"""Shared helper utilities used by both `LastToken.py` and `WordToken.py`.
This module centralises functionality that would otherwise be duplicated so
that future changes only have to be made in one place.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Sequence

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------

def get_model_and_tokenizer(model_name: str):
    """Load a Hugging-Face causal-LM and matching tokenizer on the best device.

    Parameters
    ----------
    model_name : str
        Hugging-Face model identifier, e.g. ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``.

    Returns
    -------
    tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]
        The model (fp16 on GPU when available) and tokenizer (with ``pad_token`` set).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure ``pad_token`` is defined so generation does not error.
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# -----------------------------------------------------------------------------
# Geometry / manifold analysis helpers
# -----------------------------------------------------------------------------

def analyze_manifolds(
    all_activations_by_concept: Dict[str, torch.Tensor],
    *,
    skip_insufficient_samples: bool = False,
) -> Dict[str, dict]:
    """Centre concept manifolds and compute PCA-based *effective* directions.

    For each concept's activation matrix of shape ``[N, d]`` this will:
    1. Subtract the *global* centroid shared by all concepts to remove coarse bias.
    2. Run full-rank PCA (on CPU via NumPy) to obtain eigenvectors & eigenvalues.
    3. Mark a principal component as *effective* when its variance exceeds the
       mean eigenvalue for that concept.

    Parameters
    ----------
    all_activations_by_concept : dict[str, torch.Tensor]
        Mapping concept → activation matrix (``dtype`` preserved).
    skip_insufficient_samples : bool, default False
        If True, concepts with <2 samples are ignored (PCA requirement) instead
        of raising an error.

    Returns
    -------
    dict[str, dict]
        For every concept kept, returns ``{"pca", "eigenvectors", "eigenvalues",
        "effective_mask", "centered_acts"}``.
    """
    concept_analysis: dict[str, dict] = {}

    # Step 1: global centering -------------------------------------------------
    centroids = {c: acts.mean(dim=0) for c, acts in all_activations_by_concept.items()}
    global_centroid = torch.stack(list(centroids.values())).mean(dim=0)
    centered_acts = {
        c: acts - global_centroid for c, acts in all_activations_by_concept.items()
    }

    # Step 2: PCA per concept --------------------------------------------------
    for concept, acts in centered_acts.items():
        if skip_insufficient_samples and acts.shape[0] < 2:
            print(
                f"Concept '{concept}': Not enough samples ({acts.shape[0]}) for PCA. Skipping."
            )
            continue

        pca = PCA()
        pca.fit(acts.cpu().numpy())

        eigenvectors = torch.tensor(pca.components_, dtype=acts.dtype)
        eigenvalues = torch.tensor(pca.explained_variance_, dtype=acts.dtype)

        mean_eigval = eigenvalues.mean()
        effective_mask = eigenvalues > mean_eigval

        print(
            f"Concept '{concept}': Found {effective_mask.sum()} effective eigenvectors "
            f"out of {len(eigenvectors)} (threshold: {mean_eigval:.4f})"
        )

        concept_analysis[concept] = {
            "pca": pca,
            "eigenvectors": eigenvectors,
            "eigenvalues": eigenvalues,
            "effective_mask": effective_mask,
            "centered_acts": acts,
        }

    return concept_analysis

# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def find_top_prompts(
    prompts: Sequence[str],
    centered_acts: torch.Tensor,
    direction: torch.Tensor,
    *,
    n: int = 10,
):
    """Return the *n* prompts most aligned (±) with a given direction."""
    projections = centered_acts @ direction
    _, sorted_idx = torch.sort(projections, descending=False)
    neg_idx = sorted_idx[:n]
    pos_idx = sorted_idx[-n:].flip(dims=[0])
    return {
        "positive": [prompts[i] for i in pos_idx],
        "negative": [prompts[i] for i in neg_idx],
    }

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_avg_eigenvalues(eigenvalue_data: Dict[int, float], model_name: str, prefix: str):
    """Plot average eigenvalue of a concept manifold across layers."""
    if not eigenvalue_data:
        print("No eigenvalue data to plot.")
        return

    layers = sorted(eigenvalue_data.keys())
    values = [eigenvalue_data[l] for l in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, values, marker="o")
    plt.title(f'Average "Dog" Manifold Eigenvalue vs. Layer\nModel: {model_name}')
    plt.xlabel("Model Layer")
    plt.ylabel("Average Eigenvalue")
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.6)
    fname = f"{prefix}_dog_avg_eigenvalue.png"
    plt.savefig(fname)
    print(f"Saved average eigenvalue plot to {fname}")
    plt.close()


def plot_similarity_matrix(eigenvector_data: Dict[int, torch.Tensor], model_name: str, prefix: str):
    """Heat-map of cosine similarities among top eigenvectors across layers."""
    if len(eigenvector_data) < 2:
        print("Not enough eigenvector data to create a similarity matrix.")
        return

    layers = sorted(eigenvector_data.keys())
    sims = torch.zeros((len(layers), len(layers)))
    vecs = [eigenvector_data[l] for l in layers]

    for i in range(len(layers)):
        for j in range(len(layers)):
            sims[i, j] = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()

    plt.figure(figsize=(10, 8))
    sns.heatmap(sims, annot=True, fmt=".2f", cmap="viridis", xticklabels=layers, yticklabels=layers)
    plt.title(f'Cosine Similarity of "Dog" Manifold PC0 Across Layers\nModel: {model_name}')
    plt.xlabel("Model Layer")
    plt.ylabel("Model Layer")
    fname = f"{prefix}_dog_pc0_similarity.png"
    plt.savefig(fname)
    print(f"Saved eigenvector similarity matrix to {fname}")
    plt.close()
