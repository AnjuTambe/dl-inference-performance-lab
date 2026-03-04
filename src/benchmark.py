import torch
import pandas as pd
from models.model_loader import load_model, get_dummy_input
from src.utils import measure_latency
from src.optimizations import apply_weight_only_quantization, export_torchscript

def run_benchmarks(model_name="resnet18"):
    print(f"--- Starting Benchmarks for {model_name} ---")
    
    # 0. Setup
    input_tensor = get_dummy_input(batch_size=1)
    results = []

    # 1. Baseline Model
    model = load_model(model_name)
    print(f"\n1. Benchmarking Baseline {model_name} (FP32)...")
    baseline_stats = measure_latency(model, input_tensor)
    baseline_stats["experiment"] = "Baseline (FP32)"
    results.append(baseline_stats)

    # 2. Optimized (TorchScript)
    print(f"\n2. Benchmarking TorchScript Optimized...")
    ts_model = export_torchscript(model, input_tensor)
    ts_stats = measure_latency(ts_model, input_tensor)
    ts_stats["experiment"] = "TorchScript"
    results.append(ts_stats)

    # 3. Optimized (Quantized INT8)
    # Note: Using dynamic quantization for demo
    print(f"\n3. Benchmarking Quantized (INT8)...")
    try:
        q_model = apply_weight_only_quantization(model)
        q_stats = measure_latency(q_model, input_tensor)
        q_stats["experiment"] = "Quantized (INT8)"
        results.append(q_stats)
    except Exception as e:
        print(f"Warning: Quantization failed on this hardware. Error: {e}")
        # Add a dummy entry so the dataframe doesn't break
        results.append({
            "avg_ms": 0, "std_ms": 0, "p95_ms": 0, "fps": 0, "experiment": "Quantized (FAILED)"
        })

    # Output Results
    df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(df[["experiment", "avg_ms", "fps"]])
    
    # Save to CSV
    output_path = f"results/benchmarks/{model_name}_cpu_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    run_benchmarks("resnet18")
