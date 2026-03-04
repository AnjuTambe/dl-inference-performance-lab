# Deep Learning Inference Performance Lab (CPU Optimization + Profiling)

This project demonstrates a performance-first approach to deep learning inference. It benchmarks pre-trained models (ResNet, MobileNet) on CPU and explores optimization techniques to reduce latency and increase throughput.

## Why this Project?
Performance engineering for deep learning involves:
- **Baseline Measurements**: Establishing a clear latency/throughput standard.
- **Optimization**: Using techniques like **TorchScript** and **Quantization (INT8)**.
- **Profiling**: Identifying bottlenecks with the **PyTorch Profiler**.
- **Analysis**: Proving improvements with data.

## Project Structure
- `src/benchmark.py`: Main script for latency/throughput tests.
- `src/optimizations.py`: Helpers for TorchScript and INT8 quantization.
- `src/profiler_run.py`: Captures execution traces and per-operator CPU time.
- `results/`: Contains benchmarking CSVs and profiling reports.

## Getting Started
1. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
2. **Run Benchmarks**:
   ```bash
   python3 -m src.benchmark
   ```
3. **Run Profiler**:
   ```bash
   python3 -m src.profiler_run
   ```

## Optimization Techniques
- **TorchScript**: Just-In-Time (JIT) compilation for faster execution.
- **Quantization (INT8)**: Reducing weight precision from 32-bit to 8-bit to speed up CPU inference.

## Sample Results (ResNet18 on CPU)
| Experiment | Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- |
| Baseline (FP32) | ~X ms | ~Y fps |
| TorchScript | ~X ms | ~Y fps |
| Quantized (INT8) | ~X ms | ~Y fps |

*(Actual results will depend on your specific hardware.)*
