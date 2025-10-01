"""
Benchmark script to compare CPU vs GPU performance for the static example
"""
import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the necessary components
from static_example import f_nn, T_map, init_weights, train

def run_benchmark(device_name, num_iterations=1000):
    """Run a single benchmark on specified device"""
    device = torch.device(device_name)
    print(f"\n{'='*60}")
    print(f"Benchmarking on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")
    
    # Setup parameters (smaller than full run for faster benchmark)
    N = 1000
    L = 2
    dy = 2
    num_neurons = 32
    BATCH_SIZE = 128
    LR = 1e-3
    ITERS = num_iterations
    sigma_w = 0.4
    
    # Generate data on device
    X = torch.randn(N, L, device=device)
    def h(x):
        return 0.5 * x * x
    Y = h(X).view(-1, dy) + sigma_w * torch.randn(N, dy, device=device)
    
    # Initialize models on device
    f = f_nn(L, dy, num_neurons).to(device)
    T = T_map(L, dy, num_neurons).to(device)
    f.apply(init_weights)
    T.apply(init_weights)
    
    # Warm-up (important for GPU)
    print("Warming up...")
    for _ in range(10):
        idx = torch.randperm(X.shape[0], device=device)[:BATCH_SIZE]
        X_batch = X[idx]
        Y_batch = Y[idx]
        _ = T.forward(X_batch, Y_batch)
        _ = f.forward(X_batch, Y_batch)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual benchmark
    print(f"Running {ITERS} iterations...")
    start_time = time.time()
    
    train(f, T, X, Y, ITERS, LR, BATCH_SIZE)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Ensure all GPU operations complete
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Results for {device}:")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Time per iteration: {elapsed_time/ITERS*1000:.2f} ms")
    print(f"{'='*60}\n")
    
    return elapsed_time

def main():
    print("\n" + "="*60)
    print("CPU vs GPU Benchmark for Optimal Transport Filtering")
    print("="*60)
    
    # Check CUDA availability
    has_cuda = torch.cuda.is_available()
    print(f"\nCUDA Available: {has_cuda}")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get number of iterations from command line or use default
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"Number of iterations: {num_iterations}")
    
    # Run CPU benchmark
    cpu_time = run_benchmark('cpu', num_iterations)
    
    # Run GPU benchmark if available
    if has_cuda:
        gpu_time = run_benchmark('cuda', num_iterations)
        
        # Compare results
        speedup = cpu_time / gpu_time
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"CPU Time:  {cpu_time:.2f} seconds")
        print(f"GPU Time:  {gpu_time:.2f} seconds")
        print(f"Speedup:   {speedup:.2f}x faster on GPU")
        print(f"Time saved: {cpu_time - gpu_time:.2f} seconds")
        print("="*60 + "\n")
    else:
        print("\nGPU not available. Only CPU benchmark completed.")

if __name__ == "__main__":
    main()
