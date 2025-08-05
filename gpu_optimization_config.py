"""
GPU Optimization Configuration for PlayDiffusion RVC
This file contains settings to optimize GPU usage and reduce CPU bottlenecks.
"""

import torch
import os

# GPU Memory Management Settings
GPU_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory
GPU_MEMORY_GROWTH = True   # Allow GPU memory to grow as needed

# Threading and Processing Settings
MAX_WORKERS = 4            # Maximum number of worker threads for I/O operations
CHUNK_SIZE = 1024 * 1024   # 1MB chunks for file operations

# Audio Processing Settings
AUDIO_CHUNK_SIZE = 16000   # Process audio in 1-second chunks (16kHz)
RESAMPLE_CHUNK_SIZE = 8000  # Resample in smaller chunks to reduce memory usage

# Model Settings
MODEL_DTYPE = torch.float16  # Use half precision to reduce memory usage
BATCH_SIZE = 1              # Process one audio file at a time


def configure_gpu():
    """Configure GPU settings for optimal performance"""
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

        # Enable memory growth
        if GPU_MEMORY_GROWTH:
            torch.cuda.empty_cache()

        # Set default tensor type to half precision
        torch.set_default_dtype(torch.float16)

        print(f"GPU configured: {torch.cuda.get_device_name()}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")


def get_optimal_device():
    """Get the optimal device for processing"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Environment variables for optimization
# Optimize for common GPU architectures
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5;8.0;8.6')
# Disable synchronous CUDA operations
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('OMP_NUM_THREADS', str(
    MAX_WORKERS))  # Limit OpenMP threads
