#!/usr/bin/env python3
"""
Simple CUDA environment diagnostic script
"""

import torch
import os

print("=== CUDA Environment Diagnostic ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Test CUDA functionality
    try:
        x = torch.tensor([1.0, 2.0]).cuda()
        y = x * 2
        print(f"CUDA test successful: {y}")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"CUDA test failed: {e}")
else:
    print("CUDA not available")

# Check environment variables
print("\n=== Environment Variables ===")
cuda_vars = [k for k in os.environ.keys() if 'CUDA' in k]
for var in cuda_vars:
    print(f"{var}: {os.environ[var]}")

print("\n=== Recommendations ===")
if not torch.cuda.is_available():
    print("1. Check CUDA installation")
    print("2. Check PyTorch CUDA compatibility")
else:
    print("1. Restart Python session if CUDA_VISIBLE_DEVICES was changed")
    print("2. Clear GPU memory: nvidia-smi")
    print("3. Check for competing processes")
