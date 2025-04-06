import torch
from torch import nn
import torchinfo
from thop import profile
from fvcore.nn import FlopCountAnalysis
from models.discriminator import Discriminator
from models.generator import Generator
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model, input_size=(1, 3, 224, 224)):
    """
    Get detailed information about a model including parameters and shape.

    Args:
        model: PyTorch model
        input_size: Input dimensions (batch_size, channels, height, width)

    Returns:
        Model summary from torchinfo
    """
    return torchinfo.summary(model, input_size=input_size, verbose=0)


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """
    Calculate FLOPs for a PyTorch model.

    Args:
        model: PyTorch model
        input_size: Input dimensions (batch_size, channels, height, width)

    Returns:
        flops: Number of floating point operations
        params: Number of parameters
    """
    # Create a random input tensor
    model.to(device)
    input_tensor = torch.randn(input_size).to(device)

    # Method 1: Using thop
    flops, params = profile(model, inputs=(input_tensor,))

    # Method 2: Using fvcore (more accurate for complex models)
    flops_fvcore = FlopCountAnalysis(model, input_tensor).total()

    return {"thop_flops": flops, "fvcore_flops": flops_fvcore, "params": params}


def print_model_efficiency(model, input_size=(1, 3, 224, 224)):
    """
    Print comprehensive efficiency metrics for a model.

    Args:
        model: PyTorch model
        input_size: Input dimensions (batch_size, channels, height, width)
    """
    # Get parameter count
    model.to(device)
    param_count = count_parameters(model)
    print(f"Trainable parameters: {param_count:,}")

    # Get FLOPs
    efficiency_metrics = calculate_flops(model, input_size)

    # Print results
    print(f"FLOPs (thop): {efficiency_metrics['thop_flops']:,}")
    print(f"FLOPs (fvcore): {efficiency_metrics['fvcore_flops']:,}")

    # Get detailed model summary
    print("\nDetailed Model Summary:")
    summary = get_model_info(model, input_size)
    print(summary)


# Example usage
if __name__ == "__main__":
    
    discriminator = Discriminator()
    generator = Generator()
    
    # dsconvdiscriminator = DSConvDiscriminator()
    # dsconvgenerator = DSConvGenerator()
    
    # discriminator = Discriminator()
    # transformer_generator = GeneratorWithTransformer()

    print("Discriminator")
    print_model_efficiency(discriminator)
    print("-" * 100)
    print("Generator")
    print_model_efficiency(generator)
