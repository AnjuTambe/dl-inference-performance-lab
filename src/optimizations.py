import torch
import torch.quantization

# Set quantization engine for macOS (ARM64)
if 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'

def apply_weight_only_quantization(model):
    """
    Apply simple dynamic quantization (weight-only) which is 
    effective for CPU inference.
    """
    print("Applying dynamic quantization to the model...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    return quantized_model

def export_torchscript(model, example_input):
    """
    Exports the model to TorchScript (JIT).
    """
    print("Exporting model to TorchScript...")
    traced_model = torch.jit.trace(model, example_input)
    return traced_model
