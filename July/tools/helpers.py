"""Core utilities for transformer concept manifold analysis."""

from __future__ import annotations
from typing import Dict, Sequence, Optional, Union, Any, Tuple, List
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = False

def ensure_tensor_compatibility(tensor: torch.Tensor, target_device: Union[str, torch.device], target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ensure tensor is on correct device with correct dtype for numerical stability."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")
    
    # Convert to target dtype and device
    tensor = tensor.to(device=target_device, dtype=target_dtype)
    
    # Check for NaN or infinite values
    if torch.isnan(tensor).any():
        print("Warning: NaN values detected in tensor")
    if torch.isinf(tensor).any():
        print("Warning: Infinite values detected in tensor")
    
    return tensor

def get_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with optimal device placement and caching."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name}")
    print("Attempting to use local cache to avoid HuggingFace API rate limits...")
    
    try:
        # Try loading with local_files_only first (uses cache, avoids API calls)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            device_map=device,
            local_files_only=True  # Use only cached files
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✓ Successfully loaded from local cache")
        
    except Exception as e:
        print(f"Local cache failed: {e}")
        print("Falling back to online download (may hit rate limits)...")
        
        # Fallback to normal loading with better error handling
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype, 
                device_map=device,
                resume_download=True,  # Resume interrupted downloads
                force_download=False   # Don't force redownload if cached
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                resume_download=True,
                force_download=False
            )
            print("✓ Successfully loaded from HuggingFace Hub")
            
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            print("Solutions:")
            print("1. Wait a few minutes and try again (rate limit)")
            print("2. Set up HuggingFace authentication: huggingface-cli login")
            print("3. Download model manually first: huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct")
            raise e2
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def analyse_manifolds(all_activations_by_concept: Dict[str, torch.Tensor], local_centre: bool = False) -> Dict[str, Dict[str, Any]]:
    """Compute PCA manifolds with configurable centering strategy.
    
    Args:
        all_activations_by_concept: Dictionary mapping concept names to activation tensors
        local_centre: If True, use local centering for perturbations (concept-specific reference frame).
                     If False, use global centering for perturbations (cross-concept reference frame).
                     
    Note: Individual concept PCA is ALWAYS computed on concept-centered data regardless of this flag.
          The local_centre flag only affects how perturbations are applied later.
    """
    concept_analysis: Dict[str, Dict[str, Any]] = {}
    device = torch.device('cpu')
    
    # Compute centroids for each concept
    centroids = {c: acts.to(device).mean(dim=0) for c, acts in all_activations_by_concept.items()}
    
    # Compute global centroid for perturbation reference (regardless of centering mode)
    global_centroid = torch.stack(list(centroids.values())).mean(dim=0)
    
    if local_centre:
        print("Using LOCAL centering mode: Perturbations will use concept-specific reference frames")
    else:
        print("Using GLOBAL centering mode: Perturbations will use global reference frame")
    
    print("Individual concept PCA: Each concept centered by its own mean (always)")
    
    # ALWAYS center each concept by its own mean for PCA computation
    # This ensures each concept's principal components capture variation within that concept
    centered_acts = {
        c: acts.to(device) - centroids[c] for c, acts in all_activations_by_concept.items()
    }

    for concept, acts in centered_acts.items():
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
            "centroid": centroids[concept],
            "global_centroid": global_centroid,
            "local_centre": local_centre
        }

    return concept_analysis

def find_top_prompts(
    prompts: Sequence[str],
    centered_acts: torch.Tensor,
    direction: torch.Tensor,
    n: int = 10,
    use_normalized_projection: bool = False,
) -> Dict[str, List[str]]:
    """Find prompts with extreme projections along specified direction."""
    if len(prompts) != centered_acts.shape[0]:
        raise ValueError(f"Prompt count mismatch: {len(prompts)} != {centered_acts.shape[0]}")
    
    projections = centered_acts @ direction
    
    if use_normalized_projection:
        vector_magnitudes = torch.norm(centered_acts, dim=1)
        vector_magnitudes = torch.clamp(vector_magnitudes, min=1e-8)
        values_to_sort = projections / vector_magnitudes
        value_name = "normalized projection"
    else:
        values_to_sort = projections
        value_name = "projection"
    
    _, sorted_idx = torch.sort(values_to_sort, descending=False)
    neg_idx = sorted_idx[:min(n, len(sorted_idx))]
    pos_idx = sorted_idx[-min(n, len(sorted_idx)):].flip(dims=[0])
    
    # Debug output
    print(f"\n{value_name.capitalize()} values:")
    for i in range(min(3, len(pos_idx))):
        if use_normalized_projection:
            print(f"Top positive #{i+1}: {values_to_sort[pos_idx[i]]:.4f} (raw: {projections[pos_idx[i]]:.4f})")
        else:
            print(f"Top positive #{i+1}: {values_to_sort[pos_idx[i]]:.4f}")
    for i in range(min(3, len(neg_idx))):
        if use_normalized_projection:
            print(f"Top negative #{i+1}: {values_to_sort[neg_idx[i]]:.4f} (raw: {projections[neg_idx[i]]:.4f})")
        else:
            print(f"Top negative #{i+1}: {values_to_sort[neg_idx[i]]:.4f}")
    
    return {
        "positive": [prompts[i.item()] for i in pos_idx],
        "negative": [prompts[i.item()] for i in neg_idx],
    }

def plot_avg_eigenvalues(eigenvalue_data: Dict[int, float], model_name: str, prefix: str) -> None:
    """Plot eigenvalue evolution across layers."""
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
    print(f"Saved plot: {fname}")
    plt.close()

def plot_similarity_matrix(eigenvector_data: Dict[int, torch.Tensor], model_name: str, prefix: str) -> None:
    """Generate cosine similarity heatmap across layers."""
    if len(eigenvector_data) < 2:
        print("Insufficient data for similarity matrix.")
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
    print(f"Saved plot: {fname}")
    plt.close()

def generate_with_perturbation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, str]]],
    layer_idx: int,
    direction: torch.Tensor,
    magnitude: float,
    eigenvalue: torch.Tensor,
    target_token_idx: Optional[int] = None,
    perturb_once: bool = False
) -> str:
    """Generate text with activation perturbation at specified layer."""
    perturbation = direction * magnitude * torch.sqrt(eigenvalue)
    
    # Normalize input format
    if isinstance(inputs, dict):
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        prompt_length = input_ids.shape[1]
    elif isinstance(inputs, list):
        input_ids = tokenizer.apply_chat_template(
            inputs, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        attention_mask = None
        prompt_length = input_ids.shape[1]
    else:
        input_ids = inputs
        attention_mask = None
        prompt_length = inputs.shape[1]
    
    if target_token_idx is None:
        target_token_idx = -1
    
    def hook_fn_modify(module: torch.nn.Module, input: Any, output: Any) -> Any:
        hidden_states = output[0]
        if perturb_once:
            if hidden_states.shape[1] > 1:
                if target_token_idx == -1:
                    hidden_states[:, -1, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
                else:
                    hidden_states[:, target_token_idx, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        else:
            if target_token_idx == -1:
                hidden_states[:, -1, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
            else:
                hidden_states[:, target_token_idx, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        
        return (hidden_states,) + output[1:]

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_modify)
    
    try:
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
        
        prompt_length = inputs.shape[1] if hasattr(inputs, 'shape') else input_ids.shape[1]
        decoded_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
        return decoded_text
    finally:
        # Ensure hook is always removed, even if an exception occurs
        hook_handle.remove()

def run_perturbation_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: Union[List[Dict[str, str]], torch.Tensor],
    layer_idx: int,
    concept_analysis: Dict[str, Any],
    target_concept: str,
    target_token_idx: Optional[int] = None,
    perturb_once: bool = False,
    orthogonal_mode: bool = False,
    use_largest_eigenvalue: bool = True
) -> str:
    """Execute perturbation experiment on first PC axis or orthogonal direction."""
    perturbation_scales = [-100.0, -20.0, -10.0, -5.0, -2.5, -1.5, 0.0, 1.5, 2.5, 5.0, 10.0, 20.0, 100.0]
    
    system_prompt = messages[0]['content'] if messages[0]['role'] == 'system' else ""
    user_prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
    
    if not isinstance(messages, torch.Tensor):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = messages
    
    if orthogonal_mode:
        print("\n" + "="*80)
        print(f"--- ORTHOGONAL PERTURBATION: '{target_concept}' ---")
        print(f"--- LAYER: {layer_idx} ---")
        print("="*80)
        
        # Use the PC with the lowest eigenvalue (last PC since they're sorted descending)
        orthogonal_pc_index = len(concept_analysis["eigenvalues"]) - 1
        
        direction = concept_analysis["eigenvectors"][orthogonal_pc_index].to(DEVICE)
        eigenvalue = concept_analysis["eigenvalues"][0] if use_largest_eigenvalue else concept_analysis["eigenvalues"][orthogonal_pc_index]
        
        print(f"Using PC with lowest eigenvalue: PC{orthogonal_pc_index}")
        print(f"Eigenvalue: {concept_analysis['eigenvalues'][orthogonal_pc_index].item():.6f}")
        
    else:
        # Only use the first principal component (PC0)
        axis = 0
        print("\n" + "="*80)
        print(f"--- INTERVENTION: '{target_concept}' ---")
        print(f"--- LAYER: {layer_idx}, AXIS: {axis} ---")
        print("="*80)
        
        direction = concept_analysis["eigenvectors"][axis].to(DEVICE)
        eigenvalue = concept_analysis["eigenvalues"][axis]
    
    print(f"\nSystem: '{system_prompt}'")
    print(f"User: '{user_prompt}'")

    original_output = generate_with_perturbation(
        model, tokenizer, inputs, layer_idx, direction, 0, 
        eigenvalue, target_token_idx, perturb_once
    )
    print(f"Original: {original_output}")
    
    perturbation_type = 'orthogonal direction' if orthogonal_mode else f"PC0"
    token_type = 'final token' if target_token_idx is None else 'concept token'
    print(f"\n--- Perturbing {token_type} along {perturbation_type} ---")
    
    for scale in perturbation_scales:
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, layer_idx, 
            direction, scale, eigenvalue,
            target_token_idx, perturb_once
        )
        print(f"Scale {scale:+.1f}x: {perturbed_output}")
        
    print("="*80)
        
    return original_output

def run_ablation_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: Union[List[Dict[str, str]], torch.Tensor],
    layer_idx: int,
    concept_analysis: Dict[str, Any],
    target_concept: str,
    target_token_idx: Optional[int] = None,
    perturb_once: bool = False
) -> None:
    """Execute systematic PC ablation experiments."""
    system_prompt = messages[0]['content'] if isinstance(messages, list) and messages[0].get('role') == 'system' else ""
    user_prompt = next((msg['content'] for msg in messages if isinstance(messages, list) and msg.get('role') == 'user'), "") if isinstance(messages, list) else ""
    
    if not isinstance(messages, torch.Tensor):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = messages
    
    all_eigenvectors = [evec.to(DEVICE) for evec in concept_analysis["eigenvectors"]]
    centroid = concept_analysis["centroid"].to(DEVICE)
    
    print("\n" + "="*80)
    print(f"--- ABLATION EXPERIMENT: '{target_concept}' ---")
    print(f"--- LAYER: {layer_idx} ---")
    print("="*80)
    
    print(f"\nSystem: '{system_prompt}'")
    print(f"User: '{user_prompt}'")
    
    ablation_scenarios = [
        ("All PCs (centroid only)", []),
        ("All except largest (PC0 only)", [0]),
        ("All except top two (PC0+PC1)", [0, 1]),
        ("Top 1 PC ablated", list(range(1, len(all_eigenvectors)))),
        ("Top 2 PCs ablated", list(range(2, len(all_eigenvectors)))),
        ("Top 3 PCs ablated", list(range(3, len(all_eigenvectors)))),
    ]
    
    for scenario_name, keep_pcs in ablation_scenarios:
        def create_ablation_hook(scenario_name: str, keep_pcs: List[int]):
            def ablation_hook(module: torch.nn.Module, input_tensor: Any, output: Any) -> Any:
                hidden_states = output[0]
                token_idx = target_token_idx if target_token_idx is not None else -1
                token_activation = hidden_states[0, token_idx].detach()
                
                coefficients = [torch.dot(token_activation, evec) for evec in all_eigenvectors]
                reconstructed_activation = centroid.clone()
                
                for pc_idx in keep_pcs:
                    if pc_idx < len(coefficients):
                        reconstructed_activation += coefficients[pc_idx] * all_eigenvectors[pc_idx]
                
                modified_hidden_states = hidden_states.clone()
                modified_hidden_states[0, token_idx] = reconstructed_activation
                return (modified_hidden_states,) + output[1:]
            
            return ablation_hook
        
        hook_handle = model.model.layers[layer_idx].register_forward_hook(create_ablation_hook(scenario_name, keep_pcs))
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    inputs, max_new_tokens=30, temperature=0.7, top_p=0.9, 
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id
                )
            
            prompt_length = inputs.shape[1]
            decoded_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
            print(f"{scenario_name}: {decoded_text}")
        except Exception as e:
            print(f"{scenario_name}: ERROR - {e}")
        finally:
            # Ensure hook is always removed
            hook_handle.remove()
    
    print("="*80)
