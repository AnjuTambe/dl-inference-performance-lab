import torch
import pandas as pd
from models.model_loader import load_model, get_dummy_input
from src.utils import measure_latency
import matplotlib.pyplot as plt

def run_batch_sweep(model_name="resnet18"):
    print(f"--- Starting Batch Size Sweep for {model_name} ---")
    
    model = load_model(model_name)
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []

    for bs in batch_sizes:
        print(f"\nBenchmarking Batch Size: {bs}")
        input_tensor = get_dummy_input(batch_size=bs)
        stats = measure_latency(model, input_tensor, num_iterations=50)
        
        # Calculate aggregate throughput (images per sec)
        # fps from measure_latency is batches per sec, so multiply by bs
        images_per_sec = stats["fps"] * bs
        
        results.append({
            "batch_size": bs,
            "latency_ms": stats["avg_ms"],
            "throughput_fps": images_per_sec
        })

    df = pd.DataFrame(results)
    
    # Save results
    output_path = f"results/benchmarks/{model_name}_batch_sweep.csv"
    df.to_csv(output_path, index=False)
    print(f"\nBatch sweep results saved to {output_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['batch_size'], df['throughput_fps'], marker='o', linestyle='-', color='dodgerblue')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (Images/Sec)')
    plt.title(f'Throughput vs Batch Size: {model_name}')
    plt.grid(True)
    
    plot_out = f"results/reports/batch_sweep_{model_name}.png"
    plt.savefig(plot_out)
    print(f"Batch sweep plot saved to {plot_out}")

if __name__ == "__main__":
    run_batch_sweep("resnet18")
