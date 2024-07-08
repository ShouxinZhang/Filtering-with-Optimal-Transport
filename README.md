# Nonlinear Filtering with Brenier Optimal Transport Maps

This repository is by Mohammad Al-Jarrah, Niyizhen Jin, [Bamdad Hosseini](https://bamdadhosseini.org/), [Amirhossein Taghvaei](https://www.aa.washington.edu/facultyfinder/amir-taghvaei) and contains the Python source code to reproduce the experiments in our 2024 paper [Nonlinear Filtering with Brenier Optimal Transport Maps](https://openreview.net/forum?id=blzDxD6bKt). 


We perform several numerical experiments to study the performance of the optimal transport (OT) approach in comparison with the ensemble Kalman filte (EnKF) and sequential importance resampling (SIR) algorithms.  The OT algorithm consists of solving a max-min problem over function classes $f\in {\cal F}, T\in \cal T$ with $f, T$ 
taken to be resdiual neural nets. The network weights are learned with a gradient ascent-descent procedure using the Adam optimization algorithm. To reduce the computational cost, the optimization iteration number decreases as the time grows because the OT map is not expected to change significantly from a time step to the next one. Next we present some results from the paper.

## 1. A bimodal static example
Consider the task of computing the conditional distribution of a Gaussian hidden random variable $X \sim N(0,I_n)$ given the observation

$$
\begin{aligned} 
    Y=\frac{1}{2}X\odot X + \lambda_w W, \qquad W \sim N(0,I_n)
\end{aligned}
$$

where $\odot$ denotes the element-wise (i.e., Hadamard) product. In the following left group of figures, we set $\lambda_w=0.4$ and in the right group $\lambda_w=0.04$. 

<p align="center">
<img src="/images/squared_static_example.png" width="450" height="350"> <img src="/images/squared_static_example_high_SNR.png" width="450" height="350">
</p>


## 2. A bimodal dynamic example
We consider a dynamic version of the previous example according to the following model:

$$
\begin{aligned}
    X_{t} &= (1-\alpha) X_{t-1} + 2\lambda V_t,\quad X_0 \sim \mathcal{N}(0,I_n)\\
    Y_t &= X_t\odot X_t + \lambda W_t,
\end{aligned}
$$

where $\{V_t,W_t\}_{t=1}^\infty$ are i.i.d sequences of standard Gaussian random variables, $\alpha=0.1$ and $\lambda=\sqrt{0.1}$. The choice of $Y_t$ 
will once again lead to a bimodal posterior $\pi_t$ at every time step.

<p align="center">
<img src="/images/xx_states.png" width="250" height="250"><img src="/images/xx_mmd.png" width="250" height="250"><img src="/images/dynamic_example_d_vs_mmd.png" width="250" height="250"><img src="/images/dynamic_example_N_vs_mmd.png" width="250" height="250">
</p>


## 3. The Lorenz 63 model
We consider the three-dimensional  Lorenz 63 model which often serves as a benchmark for nonlinear filtering algorithms. The
state $X_t$ is $3$-dimensional while the observation $Y_t$ is $2$-dimensional and consists of noisy measurements of the first and third components of the state. 

<p align="center">
<img src="/images/state2_and_mse_L63.png" width="750" height="350">
</p>

## 4. The Lorenz 96 model
Consider the following Lorenz-96 model:

$$
\begin{equation}
\begin{split}
\dot{X}(k) &= (X(k+1)-X(k-2))X(k-1)-X(k)+F + \sigma V,\quad \text{for}\quad k=1,\ldots,n \\
Y_t &= \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
\end{bmatrix} X_t + \sigma W_t
\end{split}
\end{equation}
$$

for $n=9$ where $X_0 \sim \mathcal{N}(\mu_0,\sigma_0^2I_n)$ and we choose the convention that $X(-1)=X(n-1)$, $X(0) = X(n)$, and $X(n+1)=X(1)$, and $F=2$ is a forcing constant. We choose the model parameters $\mu_0 = 25\cdot 1_n$, and $\sigma_{0}^2=10^2$. The observed noise $W$ is a $n$-dimensional standard Gaussian random variable with variance equal to $1$. 


<p align="center">
<img src="/images/states_L96.png" width="450" height="350"><img src="/images/mse_L96.png" width="450" height="350">
</p>

## 5. Static image in-painting on MNIST
Consider the problem of computing conditional distributions on the $100$-dimensional latent space of generative adversarial network (GAN) trained to represent the MNIST digits. In particular, denoting the generator by $G : \mathbb{R}^{100}\rightarrow \mathbb{R}^{28\times 28}$, we consider the model:
$$Y_t = h(G(X),c_t) + \sigma W_t,\quad X\sim N(0,I_{100}),$$

where the observation function $(z,c) \in \mathbb{R}^{28\times 28} \times \mathbb R^2 \mapsto h(z,c)\in \mathbb{R}^{r\times r}$ is defined as the $r\times r$ window of pixels $z[c(1):c(1)+r,c(2):c(2)+r]$. The coordinates of the corner $c_t$ moves from left to right and top to bottom scanning a partial portion of the image called  the {\it observed part}. In order to make the problem more challenging, we fed a noisy measurement of the corner location to the algorithms. While the true image does not change, we included artificial Gaussian noise to the particles to avoid particle collapse.



<p align="center">
<img src="/images/MNIST_static_final_particles_example.png" width="1000" height="400">
</p>

<p align="center">
<img src="/images/MNIST_static_OT_particles_example.png" width="500" height="400"><img src="/images/MNIST_static_dist_particles_example.png" width="500" height="400">
</p>
## 6. Dynamic image in-painting on MNIST


Please consider reading the paper for further details on this example. Also, please consider citing our paper if you find this repository useful for your publication.

```
@inproceedings{
al-jarrah2024nonlinear,
title={Nonlinear Filtering with Brenier Optimal Transport Maps},
author={Mohammad Al-Jarrah and Niyizhen Jin and Bamdad Hosseini and Amirhossein Taghvaei},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=blzDxD6bKt}
}
```


