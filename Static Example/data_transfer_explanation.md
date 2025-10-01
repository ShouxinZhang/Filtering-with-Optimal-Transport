# CPU ↔ GPU 数据传输开销详解

## 🏗️ 硬件架构

```
┌─────────────────────────────┐      PCIe 总线       ┌─────────────────────────────┐
│         CPU 系统            │   (~16-32 GB/s)      │         GPU 系统            │
│                             │◄──────────────────►  │                             │
│  CPU 内存 (RAM)            │                      │   GPU 内存 (VRAM)          │
│  - 容量: 64GB              │                      │   - 容量: 16GB (5090)      │
│  - 带宽: ~50 GB/s          │                      │   - 带宽: ~1000 GB/s       │
│  - 延迟: 较低               │                      │   - 延迟: 极低              │
└─────────────────────────────┘                      └─────────────────────────────┘
```

## 🔄 数据传输示例

### **问题代码 (频繁传输):**

```python
# ❌ 每次迭代都传输数据
for i in range(1000):
    # 1. 在CPU上生成数据
    X_cpu = torch.randn(1000, 2)  # CPU内存
    Y_cpu = torch.randn(1000, 2)  # CPU内存
    
    # 2. 传输到GPU (耗时!)
    X_gpu = X_cpu.to('cuda')  # CPU → GPU: ~0.1ms
    Y_gpu = Y_cpu.to('cuda')  # CPU → GPU: ~0.1ms
    
    # 3. GPU计算 (快速)
    output = model(X_gpu, Y_gpu)  # GPU: 0.5ms
    loss = output.mean()
    
    # 4. 传回CPU打印 (耗时!)
    print(f"Loss: {loss.item()}")  # GPU → CPU: ~0.05ms
    
# 总时间 = 1000 × (0.1 + 0.1 + 0.5 + 0.05) = 750ms
# 其中传输占比 = (0.1 + 0.1 + 0.05) / 0.75 = 33%
```

### **优化代码 (减少传输):**

```python
# ✅ 数据保持在GPU上
# 1. 一次性传输数据到GPU
X_gpu = torch.randn(1000, 2, device='cuda')  # 直接在GPU创建
Y_gpu = torch.randn(1000, 2, device='cuda')  # 直接在GPU创建

for i in range(1000):
    # 2. 全程在GPU计算
    output = model(X_gpu, Y_gpu)  # GPU: 0.5ms
    loss = output.mean()
    
    # 3. 只在需要时才传回CPU
    if i % 100 == 0:
        print(f"Loss: {loss.item()}")  # 每100次才传一次
    
# 总时间 = 1000 × 0.5 + 10 × 0.05 = 500.5ms
# 传输占比大幅降低!
```

## 📊 实际代码中的传输点

### **1. 训练循环中的隐藏传输**

```python
# static_example.py 第150行
print("Iteration: %d/%d, loss = %.4f" %(i+1,ITERS,loss.item()))
#                                                      ↑
#                                        这里触发 GPU → CPU 传输!
```

**`loss.item()` 做了什么:**
```python
loss = torch.tensor([0.6552], device='cuda')
loss.item()  # 内部操作:
             # 1. GPU同步 (等待所有GPU操作完成)
             # 2. 从GPU内存复制到CPU内存
             # 3. 返回Python float
```

### **2. 可视化时的批量传输**

```python
# static_example.py 第238行
plt.plot(X[:,0].detach().cpu().numpy(), ...)
#                        ↑↑↑
#              这里发生大量数据传输!

# 数据流:
X (GPU) → .detach() → .cpu() → .numpy()
         保持在GPU   GPU→CPU   转为NumPy数组
                    传输整个数组!
```

## ⚡ 为什么小模型在CPU上更快?

### **GPU 计算流程开销:**

```
┌──────────────┬────────────┬──────────────┬────────────┐
│  启动kernel  │  数据传输   │   GPU计算    │  同步等待   │
│   ~0.5ms     │   ~0.2ms   │    0.5ms    │   ~0.1ms   │
└──────────────┴────────────┴──────────────┴────────────┘
总计: 1.3ms/iteration

CPU直接计算: 0.8ms/iteration  ← 更快!
```

### **GPU优势的临界点:**

| 数据规模        | Batch Size | CPU时间 | GPU时间 | 胜者 |
|----------------|-----------|---------|---------|-----|
| 小 (N=1000)    | 128       | 8ms     | 21ms    | CPU |
| 中 (N=10000)   | 512       | 80ms    | 50ms    | GPU |
| 大 (N=100000)  | 2048      | 800ms   | 100ms   | GPU |

## 🎯 优化建议

### **1. 减少 `.item()` 调用**
```python
# ❌ 每次迭代都打印
for i in range(ITERS):
    loss = compute_loss()
    print(loss.item())  # 慢!

# ✅ 累积后批量打印
losses = []
for i in range(ITERS):
    loss = compute_loss()
    if i % 100 == 0:
        losses.append(loss.detach())  # 保持在GPU
if len(losses) > 0:
    print([l.item() for l in losses])  # 批量传输
```

### **2. 使用 TensorBoard 而非实时打印**
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for i in range(ITERS):
    loss = compute_loss()
    writer.add_scalar('Loss/train', loss, i)  # 异步写入
```

### **3. 增加 Batch Size**
```python
# 小batch: GPU无法充分利用
BATCH_SIZE = 128  # 利用率: ~20%

# 大batch: GPU满载
BATCH_SIZE = 2048  # 利用率: ~80%
```

## 📈 基准测试改进建议

```python
# 当前问题: 1000次迭代太少，传输开销占主导
# 建议测试:
iterations_list = [1000, 5000, 10000, 20000]
batch_sizes = [128, 512, 2048]

for iters in iterations_list:
    for bs in batch_sizes:
        cpu_time = benchmark('cpu', iters, bs)
        gpu_time = benchmark('cuda', iters, bs)
        print(f"Iters={iters}, BS={bs}, Speedup={cpu_time/gpu_time:.2f}x")
```

## 🎓 关键要点

1. **GPU不是万能药** - 小数据量时CPU可能更快
2. **传输成本很高** - 尽量让数据留在GPU上
3. **批量处理是关键** - 增大batch size提高GPU利用率
4. **同步开销** - 每次 `.item()` 都会等待GPU完成所有操作
5. **适用场景** - GPU在大规模并行计算时才有优势

## 💻 你的RTX 5090最佳使用场景

✅ **适合GPU:**
- 大batch训练 (BS ≥ 1024)
- 完整的20000次迭代训练
- 多个实验并行运行
- 高维数据 (图像、高维状态空间)

❌ **不适合GPU (用CPU更好):**
- 小batch调试 (BS < 256)
- 少量迭代测试 (< 1000)
- 频繁打印/可视化
- 序列化操作多的代码
