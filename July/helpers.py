"""
Shared helper utilities used by both `LastToken.py` and `WordToken.py`.
"""

from __future__ import annotations
from typing import Dict, Sequence
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Common configuration
# -----------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = False

# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------

def get_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Hugging-Face causal-LM and matching tokenizer on the best device.
    Returns the model (fp16 on GPU when available) and tokenizer (with ``pad_token`` set).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# -----------------------------------------------------------------------------
# Geometry / manifold analysis helpers
# -----------------------------------------------------------------------------

def analyse_manifolds(all_activations_by_concept: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """Centre concept manifolds and compute PCA-based effective directions.

    For each concept's activation matrix of shape [N, d] this will:
    1. Subtract the global centroid shared by all concepts to remove coarse bias.
    2. Run full-rank PCA (on CPU via NumPy) to obtain eigenvectors & eigenvalues.
    3. Mark a principal component as effective when its variance exceeds the
       mean eigenvalue for that concept.

    For every concept kept, returns {"pca", "eigenvectors", "eigenvalues","effective_mask", "centered_acts"}.
    """
    concept_analysis: dict[str, dict] = {}

    device = torch.device('cpu')
    
    # Move all tensors to the same device (CPU) and compute centroids
    centroids = {c: acts.to(device).mean(dim=0) for c, acts in all_activations_by_concept.items()}
    global_centroid = torch.stack(list(centroids.values())).mean(dim=0)
    centered_acts = {
        c: acts.to(device) - global_centroid for c, acts in all_activations_by_concept.items()
    }

    for concept, acts in centered_acts.items():
        # Skip concepts with no activations (empty tensors)
        if acts.shape[0] == 0:
            print(f"Warning: Concept '{concept}' has no activations. Skipping PCA analysis for this concept.")
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
# Misc
# -----------------------------------------------------------------------------

def find_top_prompts(
    prompts: Sequence[str],
    centered_acts: torch.Tensor,
    direction: torch.Tensor,
    n: int = 10,
)-> dict[str, list[str]]:
    """Return the n prompts furthest in direction of particulat eigenvector."""
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

def plot_avg_eigenvalues(eigenvalue_data: Dict[int, float], model_name: str, prefix: str)-> None:
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


def plot_similarity_matrix(eigenvector_data: Dict[int, torch.Tensor], model_name: str, prefix: str)-> None:
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

# -----------------------------------------------------------------------------
# Perturbation utilities
# -----------------------------------------------------------------------------

def generate_with_perturbation(
    model, tokenizer, inputs, layer_idx, direction, 
    magnitude, eigenvalue, target_token_idx=None, 
    perturb_once=False
)-> str:
    """
    Generate text with a perturbation applied to a specific token's activation.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        inputs: Either a tensor of input_ids, a dict with input_ids, or a list of message dicts
        layer_idx: Layer to apply perturbation to
        direction: Direction vector for perturbation
        magnitude: Scale of perturbation
        eigenvalue: Eigenvalue to scale perturbation by
        target_token_idx: Index of token to perturb (-1 for last token if None)
        perturb_once: If True, only apply perturbation on first forward pass
    
    Returns:
        The generated text
    """
    perturbation = direction * magnitude * torch.sqrt(eigenvalue)
    
    # Process inputs to standard format
    if isinstance(inputs, dict):
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        prompt_length = input_ids.shape[1]
    elif isinstance(inputs, list):
        # Handle case where inputs is a list of message dicts
        input_ids = tokenizer.apply_chat_template(
            inputs,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        attention_mask = None
        prompt_length = input_ids.shape[1]
    else:
        # inputs is a tensor of input_ids
        input_ids = inputs
        attention_mask = None
        prompt_length = inputs.shape[1]
    
    # Default to last token if target_token_idx not specified
    if target_token_idx is None:
        target_token_idx = -1
    
    def hook_fn_modify(module, input, output)-> None:
        hidden_states = output[0]
        # Apply perturbation based on configuration
        if perturb_once:
            # Only apply during initial prompt processing (sequence length > 1)
            if hidden_states.shape[1] > 1:
                if target_token_idx == -1:
                    # Perturb last token
                    hidden_states[:, -1, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
                else:
                    # Perturb specific token
                    hidden_states[:, target_token_idx, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        else:
            # Apply on every forward pass
            if target_token_idx == -1:
                # Perturb last token
                hidden_states[:, -1, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
            else:
                # Perturb specific token
                hidden_states[:, target_token_idx, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        
        return (hidden_states,) + output[1:]

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_modify)
    
    # Prepare generation kwargs
    generate_kwargs = {
        'input_ids': input_ids,
        'max_new_tokens': 70,
        'do_sample': False,
        'pad_token_id': tokenizer.eos_token_id
    }
    if attention_mask is not None:
        generate_kwargs['attention_mask'] = attention_mask
    
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)
    
    hook_handle.remove()
    
    # Extract just the new tokens
    decoded_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    
    return decoded_text

def run_perturbation_experiment(
    model, tokenizer, messages, layer_idx, concept_analysis, 
    target_concept, test_axes=None, target_token_idx=None, 
    perturb_once=False, orthogonal_mode=False, use_largest_eigenvalue=True
)-> str:
    """
    Run perturbation experiments for a concept across multiple principal component axes
    or along the first orthogonal (ineffective) direction.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer for the model
        messages: List of message dictionaries or tensor of input_ids
        layer_idx: Layer index to apply perturbation
        concept_analysis: Analysis results for the concept
        target_concept: The name of the concept being analyzed
        test_axes: List of PC axes to test (ignored if orthogonal_mode=True)
        target_token_idx: Index of token to perturb (None for last token)
        perturb_once: If True, only apply perturbation on first forward pass
        orthogonal_mode: If True, find and use first orthogonal (ineffective) direction
        use_largest_eigenvalue: If True and in orthogonal_mode, use the largest eigenvalue 
                               for scaling instead of the orthogonal eigenvalue (which may be very small)
    """
    perturbation_scales = [-100.0, -20.0, -10.0, -5.0, -2.5, -1.5, 0.0, 1.5, 2.5, 5.0, 10.0, 20.0, 100.0]
    
    # Format for display
    system_prompt = messages[0]['content'] if messages[0]['role'] == 'system' else ""
    user_prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
    
    # Prepare inputs (if not already a tensor)
    if not isinstance(messages, torch.Tensor):
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = messages
    
    # Handle orthogonal mode - find the first ineffective direction
    if orthogonal_mode:
        print("\n" + "="*80)
        print(f"--- ORTHOGONAL PERTURBATION ON CONCEPT: '{target_concept}' ---")
        print(f"--- LAYER: {layer_idx} ---")
        print("="*80)
        
        effective_mask = concept_analysis["effective_mask"]
        
        # Find the first principal component that is *not* effective
        orthogonal_pc_index = -1
        for i, is_effective in enumerate(effective_mask):
            if not is_effective:
                orthogonal_pc_index = i
                break
        
        if orthogonal_pc_index == -1:
            print("Could not find an orthogonal (ineffective) direction to perturb.")
            return ""
            
        # Set up the direction and eigenvalue for the orthogonal case
        direction = concept_analysis["eigenvectors"][orthogonal_pc_index]
        
        # Choose which eigenvalue to use for scaling
        if use_largest_eigenvalue:
            # Use the largest eigenvalue for better scaling
            eigenvalue = concept_analysis["eigenvalues"][0]
        else:
            # Use the orthogonal direction's eigenvalue (which might be very small)
            eigenvalue = concept_analysis["eigenvalues"][orthogonal_pc_index]
            
        print(f"Perturbing along first orthogonal direction (PC{orthogonal_pc_index})...")
        # We'll use a single "axis" for the orthogonal case
        axes_to_test = [0]  # Just a dummy index since we already have the direction
        
    else:  # Regular PC perturbation mode
        if not test_axes:
            test_axes = range(5)  # Default to first 5 axes if not specified
        axes_to_test = test_axes
    
    # Run perturbation for each axis (just once for orthogonal mode)
    for i, axis in enumerate(axes_to_test):
        if not orthogonal_mode:  # For regular mode, get the direction for each axis
            print("\n" + "="*80)
            print(f"--- INTERVENTION EXPERIMENT ON CONCEPT: '{target_concept}' ---")
            print(f"--- LAYER: {layer_idx}, AXIS: {axis} ---")
            print("="*80)
            
            direction = concept_analysis["eigenvectors"][axis]
            eigenvalue = concept_analysis["eigenvalues"][axis]
        
        print(f"\nSystem Prompt: '{system_prompt}'")
        print(f"User Prompt:   '{user_prompt}'")

        original_output = generate_with_perturbation(
            model, tokenizer, inputs, layer_idx, direction, 0, 
            eigenvalue, target_token_idx, perturb_once
        )
        print(f"Original model completion: {original_output}")
        
        perturbation_type = 'orthogonal direction' if orthogonal_mode else f"PC{axis}"
        token_type = 'final token' if target_token_idx is None else 'concept token'
        print(f"\n--- Perturbing {token_type} activation along {perturbation_type} ---")
        
        for scale in perturbation_scales:
            perturbed_output = generate_with_perturbation(
                model, tokenizer, inputs, layer_idx, 
                direction, scale, eigenvalue,
                target_token_idx, perturb_once
            )
            print(f"Perturbation scale {scale:+.1f}x: {perturbed_output}")
            
        print("="*80)
        
    return original_output
