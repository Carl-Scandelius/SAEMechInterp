"""
Minimal Attention Steering Experiment

Rigorous scientific implementation for:
- Extracting final token pre-residual embeddings from Anthropic dataset
- Computing top 3 PCs per layer
- Steering model output using top 2 PCs
- Generating PC similarity matrix across layers
"""

from .core_experiment import run_experiment

__version__ = "1.0.0"
__all__ = ["run_experiment"] 