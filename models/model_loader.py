import torch
import torchvision.models as models
import ssl

# Fix for macOS SSL certificate verification error
ssl._create_default_https_context = ssl._create_unverified_context

def load_model(model_name="mobilenet_v2", pretrained=True):
    """
    Loads a pre-trained CV model from torchvision.
    Options: mobilenet_v2, resnet18, resnet50
    """
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    return model

def get_dummy_input(batch_size=1, channels=3, height=224, width=224):
    """Generates a dummy input tensor for benchmarking."""
    return torch.randn(batch_size, channels, height, width)
