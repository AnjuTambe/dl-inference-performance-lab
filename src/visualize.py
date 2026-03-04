import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_charts(model_name="resnet18"):
    csv_path = f"results/benchmarks/{model_name}_cpu_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark.py first.")
        return

    df = pd.read_csv(csv_path)

    # Set style
    plt.style.use('ggplot')
    
    # 1. Latency Chart
    plt.figure(figsize=(10, 6))
    plt.bar(df['experiment'], df['avg_ms'], color=['skyblue', 'orange', 'lightgreen'])
    plt.ylabel('Average Latency (ms)')
    plt.title(f'Inference Latency Comparison: {model_name}')
    for i, v in enumerate(df['avg_ms']):
        plt.text(i, v + 0.1, f"{v:.2f}ms", ha='center', fontweight='bold')
    
    latency_out = f"results/reports/latency_{model_name}.png"
    plt.savefig(latency_out)
    print(f"Latency chart saved to {latency_out}")

    # 2. Throughput Chart
    plt.figure(figsize=(10, 6))
    plt.bar(df['experiment'], df['fps'], color=['skyblue', 'orange', 'lightgreen'])
    plt.ylabel('Throughput (Images/Sec)')
    plt.title(f'Inference Throughput Comparison: {model_name}')
    for i, v in enumerate(df['fps']):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

    throughput_out = f"results/reports/throughput_{model_name}.png"
    plt.savefig(throughput_out)
    print(f"Throughput chart saved to {throughput_out}")

if __name__ == "__main__":
    generate_charts("resnet18")
