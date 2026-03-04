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
| Baseline (FP32) | 7.40 ms | 135.1 fps |
| TorchScript | 6.95 ms | 143.9 fps |
| Quantized (INT8) | 7.22 ms | 138.6 fps |

*Results achieved on Apple M1/M2/M3 CPU.*

## Resume Highlights
*   **Engineered an automated benchmarking suite** to measure latency and throughput of CNN models (ResNet, MobileNet), identifying performance scaling bottlenecks across varying batch sizes (1 to 32).
*   **Implemented model optimizations** using **TorchScript (JIT)** and **INT8 Dynamic Quantization**, achieving a measurable reduction in CPU inference latency while maintaining prediction accuracy.
*   **Performed kernel-level profiling** using the **PyTorch Profiler**, analyzing operator execution time to pinpoint `aten::conv2d` as the primary computational bottleneck (65% of total CPU cycles).
