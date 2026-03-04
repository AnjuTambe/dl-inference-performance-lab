import time
import torch
import numpy as np

def measure_latency(model, input_tensor, num_iterations=100, warmup_iterations=10):
    """
    Measures the average latency of model inference.
    """
    # Warm up
    print(f"Warming up for {warmup_iterations} iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)
    
    # Measure
    print(f"Benchmarking for {num_iterations} iterations...")
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Convert to ms
            
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    return {
        "avg_ms": avg_latency,
        "std_ms": std_latency,
        "p95_ms": p95_latency,
        "fps": 1000.0 / avg_latency if avg_latency > 0 else 0
    }
