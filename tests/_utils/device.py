"""Device utilities for tests."""

import torch
from typing import List


def select_most_accurate_gpu_device() -> str:
    """Select the most accurate GPU device available.
    
    Returns:
        str: Device string - "mps" if available on Apple Silicon, 
             otherwise "cuda" for NVIDIA GPUs.
    """
    return "mps" if torch.backends.mps.is_available() else "cuda"


def get_available_dtypes() -> List[torch.dtype]:
    """Get list of available dtypes for testing based on GPU backend.
    
    Returns:
        List[torch.dtype]: List of dtypes to test with. 
                          Returns [float32, float64] for CUDA,
                          [float32] for MPS (due to limited float64 support).
    """
    return [torch.float32, torch.float64] if select_most_accurate_gpu_device() == "cuda" else [torch.float32]
