"""
Fisher-guided Embedding Perturbations Experiment

Rigorous scientific implementation for:
- Computing Fisher Information Matrix at residual stream activations
- Finding top eigenvectors of Fisher matrix per sentence
- Perturbing embeddings along Fisher eigenvectors with multiple scales
- Analyzing cosine similarities and next-token prediction changes
"""

from .core_experiment import run_fisher_experiment

__version__ = "1.0.0"
__all__ = ["run_fisher_experiment"] 