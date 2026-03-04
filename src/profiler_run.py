import torch
from torch.profiler import profile, record_function, ProfilerActivity
from models.model_loader import load_model, get_dummy_input

def run_profiler(model_name="resnet18"):
    print(f"--- Running PyTorch Profiler for {model_name} ---")
    
    model = load_model(model_name)
    input_tensor = get_dummy_input(batch_size=1)
    
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(input_tensor)
            
    # Print the profile results to a file
    print("\nProfiling results (Top 10 by CPU time):")
    profile_results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    print(profile_results)
    
    output_path = f"results/profiler/profiler_table.txt"
    with open(output_path, "w") as f:
        f.write(profile_results)
    
    # Save a Chrome Trace for visualization (optional)
    prof.export_chrome_trace("results/profiler/trace.json")
    print(f"\nProfile results saved to {output_path}")
    print("Chrome trace saved to results/profiler/trace.json")

if __name__ == "__main__":
    run_profiler("resnet18")
