# Deep Kalman Filter - minimal reproducible example on a toy dataset
# ---------------------------------------------------------------
# - Nonlinear transition & emission with actions
# - q-RNN recognition network + reparameterization
# - Time-factorized ELBO with closed-form Gaussian KLs
# ---------------------------------------------------------------

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from contextlib import nullcontext
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast, GradScaler = nullcontext, (lambda enabled=False: nullcontext())
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# 1) Toy data generator
# ----------------------------
def make_toy_dataset(N=2000, T=15, z_dim=4, x_dim=6, u_dim=1,
                     trans_noise=0.15, emis_noise=0.05, device="cpu"):
    """
    Latent dynamics (ground truth): z_t = z_{t-1} + 0.4*tanh(Wz*z_{t-1} + Wu*u_{t-1}) + eps
    Emission: x_t = tanh(We*z_t) + noise
    Actions u_t in {-1, 0, +1}
    """
    Wz = torch.randn(z_dim, z_dim) / math.sqrt(z_dim)
    Wu = torch.randn(z_dim, u_dim) / math.sqrt(u_dim)
    We = torch.randn(x_dim, z_dim) / math.sqrt(z_dim)

    def step(z_prev, u_prev):
        mean = z_prev + 0.4 * torch.tanh(z_prev @ Wz.T + u_prev @ Wu.T)
        z_t = mean + trans_noise * torch.randn_like(z_prev)
        x_mean = torch.tanh(z_t @ We.T)
        x_t = x_mean + emis_noise * torch.randn(z_prev.size(0), x_dim)
        return z_t, x_t

    X = torch.zeros(N, T, x_dim)
    U = torch.zeros(N, T, u_dim)
    Z = torch.zeros(N, T, z_dim)
    # actions at t=0 are 0; from t>=1 sampled
    actions = torch.tensor([-1.0, 0.0, 1.0]).view(3,1)

    z = torch.randn(N, z_dim)
    for t in range(T):
        if t == 0:
            u_prev = torch.zeros(N, u_dim)
        else:
            idx = torch.randint(0, 3, (N,))
            u_prev = actions[idx]
        z, x = step(z, u_prev)
        X[:, t] = x
        U[:, t] = u_prev
        Z[:, t] = z

    # train / test split
    # 濮嬬粓杩斿洖 CPU 寮犻噺锛涜缁冩椂鎸夐渶鎷疯礉鍒扮洰鏍囪澶囦互瀹炵幇 CPU/GPU 骞惰娴佹按
    n_train = int(0.8 * N)
    data = {
        "train": (X[:n_train].contiguous(), U[:n_train].contiguous()),
        "test":  (X[n_train:].contiguous(), U[n_train:].contiguous()),
    }
    return data

# ----------------------------
# 2) Model components
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, layers=2, act=nn.Tanh):
        super().__init__()
        seq = []
        d = in_dim
        for _ in range(layers-1):
            seq += [nn.Linear(d, hidden), act()]
            d = hidden
        seq += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x)

class DKF(nn.Module):
    def __init__(self, x_dim, z_dim, u_dim, enc_hid=64, trans_hid=64, emis_hid=64):
        super().__init__()
        self.x_dim, self.z_dim, self.u_dim = x_dim, z_dim, u_dim

        # Recognition network q(z_t | x_{1:t}, u_{1:t}) -- q-RNN
        self.enc_rnn = nn.LSTM(input_size=x_dim + u_dim, hidden_size=enc_hid,
                               num_layers=1, batch_first=True)
        self.enc_out = nn.Linear(enc_hid, 2*z_dim)  # -> [mu_q, logvar_q]

        # Transition p(z_t | z_{t-1}, u_{t-1})
        self.trans_mean = MLP(z_dim + u_dim, z_dim, hidden=trans_hid, layers=2)
        self.trans_logvar = MLP(z_dim + u_dim, z_dim, hidden=trans_hid, layers=2)

        # Emission p(x_t | z_t)
        self.emis_mean = MLP(z_dim, x_dim, hidden=emis_hid, layers=2)
        # learn diagonal log-variance (time-invariant for simplicity)
        self.emis_logvar = nn.Parameter(torch.zeros(x_dim))

        # prior p(z1) = N(0, I)
        self.register_buffer("prior_mu", torch.zeros(z_dim))
        self.register_buffer("prior_logvar", torch.zeros(z_dim))

    @staticmethod
    def reparameterize(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5*logvar) * eps

    @staticmethod
    def kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
        # KL(q||p) for diagonal Gaussians
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1.0)
        return kl.sum(-1)

    def elbo(self, x, u):
        """
        x: [B, T, x_dim], u: [B, T, u_dim]
        Returns: elbo (scalar), recon term, KL term
        """
        B, T, _ = x.size()
        device = x.device

        # 璇嗗埆缃戠粶锛氭妸 (x_t, u_{t-1}) 浣滀负杈撳叆锛坱=0 鐨?u_{-1}=0锛?
        u_prev = torch.zeros(B, 1, self.u_dim, device=device)
        u_shift = torch.cat([u_prev, u[:, :-1, :]], dim=1)
        enc_in = torch.cat([x, u_shift], dim=-1)  # [B, T, x_dim+u_dim]
        h, _ = self.enc_rnn(enc_in)
        enc_params = self.enc_out(h)              # [B, T, 2*z_dim]
        mu_q, logvar_q = torch.chunk(enc_params, 2, dim=-1)

        # 閫愭椂鍒婚噰鏍?z_t锛屽苟绱 KL 涓庨噸鏋勯」
        kl_total = x.new_zeros(B)
        recon_total = x.new_zeros(B)
        z_prev = None

        emis_logvar = torch.clamp(self.emis_logvar, -6.0, 6.0)

        for t in range(T):
            mu_t, logvar_t = mu_q[:, t, :], torch.clamp(logvar_q[:, t, :], -8.0, 8.0)
            z_t = self.reparameterize(mu_t, logvar_t)

            # KL
            if t == 0:
                mu_p = self.prior_mu.expand_as(mu_t)
                logvar_p = self.prior_logvar.expand_as(logvar_t)
            else:
                trans_in = torch.cat([z_prev, u[:, t-1, :]], dim=-1)
                mu_p = self.trans_mean(trans_in)
                logvar_p = torch.clamp(self.trans_logvar(trans_in), -6.0, 6.0)
            kl_t = self.kl_diag_gaussians(mu_t, logvar_t, mu_p, logvar_p)
            kl_total = kl_total + kl_t

            # Reconstruction log-likelihood (diagonal Gaussian)
            x_mean = self.emis_mean(z_t)
            diff2 = (x[:, t, :] - x_mean)**2
            recon_t = -0.5 * (
                self.x_dim * math.log(2*math.pi) +
                emis_logvar.sum() +
                (diff2 / torch.exp(emis_logvar)).sum(-1)
            )
            recon_total = recon_total + recon_t

            z_prev = z_t

        elbo = recon_total - kl_total  # [B]
        # 杩斿洖骞冲潎 ELBO
        return elbo.mean(), recon_total.mean(), kl_total.mean()

    def forward(self, x, u):
        elbo, rec, kl = self.elbo(x, u)
        return {"elbo": elbo, "recon": rec, "kl": kl}

    @torch.no_grad()
    def counterfactual(self, x_prefix, u_prefix, u_future, horizon):
        """
        缁欏畾鍓嶇紑 (x_{1:t}, u_{1:t}) 浠ュ強鏈潵鍔ㄤ綔搴忓垪 u_{t+1:t+h}锛?
        鍏堟帹鏂?z_t锛岀劧鍚庡湪鐢熸垚妯″瀷涓嬪墠鍚戦噰鏍锋湭鏉ワ紝骞惰繑鍥為娴嬬殑 x_{t+1:t+h} 鐨勫潎鍊笺€?
        """
        self.eval()
        B, t, _ = x_prefix.size()
        device = x_prefix.device

        # infer q(z_1..z_t), we take the last posterior mean as z_t sample
        u_prev0 = torch.zeros(B, 1, self.u_dim, device=device)
        u_shift = torch.cat([u_prev0, u_prefix[:, :-1, :]], dim=1)
        enc_in = torch.cat([x_prefix, u_shift], dim=-1)
        h, _ = self.enc_rnn(enc_in)
        mu_q, logvar_q = torch.chunk(self.enc_out(h), 2, dim=-1)
        z_t = mu_q[:, -1, :]  # use mean for stability

        xs = []
        z_prev = z_t
        for k in range(horizon):
            u_prev = u_future[:, k, :]
            trans_in = torch.cat([z_prev, u_prev], dim=-1)
            mu_p = self.trans_mean(trans_in)
            logvar_p = torch.clamp(self.trans_logvar(trans_in), -6.0, 6.0)
            z = mu_p  # use mean trajectory; or sample with reparameterization
            x_mean = self.emis_mean(z)
            xs.append(x_mean.unsqueeze(1))
            z_prev = z
        return torch.cat(xs, dim=1)  # [B, horizon, x_dim]

    @torch.no_grad()
    def predict_future(self, x, u, pred_steps=5):
        """
        涓€姝ラ娴嬶細缁欏畾瀹屾暣搴忓垪 x[:, :T], u[:, :T]锛岄娴嬫湭鏉?pred_steps 姝?
        杩斿洖: 棰勬祴鐨?x 鍜岀湡瀹炵殑瑙傛祴鍣０鏍囧噯宸?
        """
        self.eval()
        B, T, _ = x.size()
        device = x.device
        
        # 浣跨敤鍓?T-pred_steps 姝ヤ綔涓哄巻鍙?
        t_split = T - pred_steps
        x_hist = x[:, :t_split, :]
        u_hist = u[:, :t_split, :]
        u_future = u[:, t_split:, :]
        
        # 棰勬祴
        x_pred = self.counterfactual(x_hist, u_hist, u_future, horizon=pred_steps)
        
        return x_pred, x[:, t_split:, :]  # 杩斿洖棰勬祴鍜岀湡瀹炲€?

# ----------------------------
# 3) Train & evaluate
# ----------------------------
def evaluate_prediction_accuracy(model, X, U, pred_steps=5):
    """
    璇勪及妯″瀷鐨勯娴嬪噯纭巼
    
    鎸囨爣:
    - MSE (鍧囨柟璇樊): 瓒婂皬瓒婂ソ
    - MAE (骞冲潎缁濆璇樊): 瓒婂皬瓒婂ソ  
    - R虏 (鍐冲畾绯绘暟): 瓒婃帴杩?瓒婂ソ锛岃礋鍊艰〃绀烘瘮鍧囧€奸娴嬭繕宸?
    """
    model.eval()
    # 灏嗘暟鎹Щ鍔ㄥ埌涓庢ā鍨嬬浉鍚岀殑璁惧锛岄伩鍏嶅浣欑殑璁惧寰€杩?    dev = next(model.parameters()).device
    X = X.to(dev)
    U = U.to(dev)
    x_pred, x_true = model.predict_future(X, U, pred_steps=pred_steps)
    
    # 璁＄畻鍚勭鎸囨爣
    mse = torch.mean((x_pred - x_true) ** 2).item()
    mae = torch.mean(torch.abs(x_pred - x_true)).item()
    rmse = math.sqrt(mse)
    
    # R虏 score: 1 - (SS_res / SS_tot)
    ss_res = torch.sum((x_true - x_pred) ** 2).item()
    ss_tot = torch.sum((x_true - x_true.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')
    
    # 鐩稿璇樊 (鐧惧垎姣?
    relative_error = mae / (torch.abs(x_true).mean().item() + 1e-8) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Relative_Error_%': relative_error
    }

def train_dkf(epochs=200, batch_size=64, lr=1e-3, device="cpu", use_amp=True, num_workers=None):
    # 鏁版嵁闆嗕繚鎸?CPU锛岃缁冩椂鎸夐渶寮傛鎷疯礉鍒拌澶囷紝瀹炵幇 CPU/GPU 骞惰娴佹按
    data = make_toy_dataset(device="cpu")
    Xtr, Utr = data["train"]
    Xte, Ute = data["test"]

    # DataLoader 骞惰涓庡浐瀹氬唴瀛橈紙浠?CUDA 鏈夋晥锛?    if num_workers is None:
        try:
            cpu_cnt = os.cpu_count() or 4
        except Exception:
            cpu_cnt = 4
        num_workers = min(8, max(0, cpu_cnt - 1))

    pin = (device == "cuda")
    train_loader = DataLoader(
        TensorDataset(Xtr, Utr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    # cuDNN 浼樺寲锛圠STM 浼氬彈鐩婏級
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = DKF(x_dim=Xtr.size(-1), z_dim=8, u_dim=Utr.size(-1)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device == "cuda" and use_amp))

    for ep in range(1, epochs+1):
        model.train()
        elbo_sum = 0.0
        for xb, ub in train_loader:
            # 寮傛鎷疯礉鍒?GPU锛屼笌璁＄畻閲嶅彔锛圕PU/GPU 骞惰锛?            if device == "cuda":
                xb = xb.pin_memory().to(device, non_blocking=True)
                ub = ub.pin_memory().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if device == "cuda" and use_amp:
                with autocast():
                    out = model(xb, ub)
                    loss = -out["elbo"]
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                out = model(xb, ub)
                loss = -out["elbo"]
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            elbo_sum += out["elbo"].item() * xb.size(0)
        avg_elbo = elbo_sum / Xtr.size(0)

        if ep % 20 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                out_te = model(Xte.to(device), Ute.to(device))
            print(f"[{ep:03d}] train ELBO {avg_elbo:.3f} | test ELBO {out_te['elbo'].item():.3f} "
                  f"(recon {out_te['recon'].item():.3f}, KL {out_te['kl'].item():.3f})")

    # ========== 璇勪及棰勬祴鍑嗙‘鐜?==========
    print("\n" + "="*60)
    print("棰勬祴鍑嗙‘鐜囪瘎浼帮紙鏈€鍚?5 姝ラ娴嬶級")
    print("="*60)
    
    with torch.no_grad():
        # 璁粌闆嗚瘎浼?
        metrics_train = evaluate_prediction_accuracy(model, Xtr, Utr, pred_steps=5)
        print("\n銆愯缁冮泦銆?)
        for k, v in metrics_train.items():
            print(f"  {k:20s}: {v:.6f}")
        
        # 娴嬭瘯闆嗚瘎浼?
        metrics_test = evaluate_prediction_accuracy(model, Xte, Ute, pred_steps=5)
        print("\n銆愭祴璇曢泦銆?)
        for k, v in metrics_test.items():
            print(f"  {k:20s}: {v:.6f}")
    
    print("\n" + "="*60)
    print("鎸囨爣璇存槑:")
    print("  - MSE/RMSE/MAE: 瓒婂皬瓒婂ソ锛堟帴杩?琛ㄧず棰勬祴瀹岀編锛?)
    print("  - R虏: 瓒婃帴杩?瓒婂ソ锛?0.9浼樼, <0琛ㄧず姣斿潎鍊艰繕宸級")
    print("  - Relative_Error: 鐩稿璇樊鐧惧垎姣旓紙<5%浼樼锛?)
    print("="*60 + "\n")

    # counterfactual demo on 1 test sequence:
    with torch.no_grad():
        x0 = Xte[:1, :8, :].to(device)   # prefix length t=8
        u0 = Ute[:1, :8, :].to(device)
        horizon = 7
        # scenario A: keep actions as 0
        uA = torch.zeros(1, horizon, Ute.size(-1), device=device)
        # scenario B: push positive action
        uB = torch.ones(1, horizon, Ute.size(-1), device=device)
        xA = model.counterfactual(x0, u0, uA, horizon=horizon)
        xB = model.counterfactual(x0, u0, uB, horizon=horizon)
        print("Counterfactual demo: mean of last-step x under two action policies")
        print("no-action policy: ", xA[0, -1].cpu().numpy())
        print("positive-action:  ", xB[0, -1].cpu().numpy())

def _benchmark_device_once(device: str, steps: int = 8, batch_size: int = 128, use_amp=True):
    """粗略基准：在指定设备上做若干步前向+反向，返回耗时（秒）。"""
    data = make_toy_dataset(N=1024)
    X, U = data["train"]
    model = DKF(x_dim=X.size(-1), z_dim=8, u_dim=U.size(-1)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=(device == "cuda" and use_amp))

    loader = DataLoader(TensorDataset(X, U), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device=="cuda"))

    # warmup
    it = iter(loader)
    for _ in range(2):
        xb, ub = next(it)
        if device == "cuda":
            xb = xb.pin_memory().to(device, non_blocking=True)
            ub = ub.pin_memory().to(device, non_blocking=True)
        out = model(xb, ub)
        loss = -out["elbo"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    it = iter(loader)
    for _ in range(steps):
        xb, ub = next(it)
        if device == "cuda":
            xb = xb.pin_memory().to(device, non_blocking=True)
            ub = ub.pin_memory().to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if device == "cuda" and use_amp:
            with autocast():
                out = model(xb, ub)
                loss = -out["elbo"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(xb, ub)
            loss = -out["elbo"]
            loss.backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0


def auto_select_device(use_amp=True):
    """
    自动选择最快设备：
    - 若无 CUDA，则返回 CPU。
    - 若有 CUDA，则在 CPU 与 CUDA 上做一次小型基准，返回更快者。
    可通过环境变量 `DKF_DEVICE=cpu/cuda` 强制指定。
    """
    forced = os.environ.get("DKF_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced if (forced == "cpu" or torch.cuda.is_available()) else "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    try:
        t_cpu = _benchmark_device_once("cpu", steps=6, batch_size=128, use_amp=False)
        t_gpu = _benchmark_device_once("cuda", steps=6, batch_size=128, use_amp=use_amp)
        return "cuda" if t_gpu < t_cpu else "cpu"
    except Exception:
        return "cuda"

if __name__ == "__main__":
    # 鑷姩妫€娴嬪苟浣跨敤 GPU
    device = auto_select_device(use_amp=True)
    print("="*60)
    print(f"馃殌 浣跨敤璁惧: {device.upper()}")
    if device == "cuda":
        print(f"   GPU 鍨嬪彿: {torch.cuda.get_device_name(0)}")
        print(f"   鏄惧瓨瀹归噺: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   鈿狅笍  鏈娴嬪埌 CUDA锛屼娇鐢?CPU 璁粌")
    print("="*60 + "\n")
    
    train_dkf(epochs=200, batch_size=64, lr=1e-3, device=device, use_amp=True)

