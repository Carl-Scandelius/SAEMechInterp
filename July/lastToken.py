"""
Word manifold exploration.

Pass concept-labelled prompts (in QA format) into Llama. Extract corresponding token representation
in residual stream of layer l.

We now have some sample of the representation space. Ideally this is in the form of concept manifolds. 
Centre this representation to remove correlation between centroids Find 'effective' eigenvectors of the manifolds.
 Project work token (from new labelled prompt) representation onto respective 'effective' concept manifold.

Check decoded sentence for:
1) original prompt
2) original with final token embedding perturbed in eigenvec direction

Find prompts in initial dataset that are furthest in the direction of eigenvec: how are they correlated?
Is this a global feature or just for that manifold's eigenvec?

"""

import torch
from tqdm import tqdm
import gc
import json
from helpers import (
    get_model_and_tokenizer,
    analyse_manifolds,
    find_top_prompts,
    plot_avg_eigenvalues,
    plot_similarity_matrix,
    run_perturbation_experiment,
    run_projection_based_perturbation,
    run_ablation_experiment,
    MODEL_NAME,
    DEVICE,
)
from transformers import logging
logging.set_verbosity(40)

USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = False
USE_NORMALIZED_PROJECTION = True  # If True, use (projection magnitude / vector magnitude) ratio
RUN_GLOBAL_PC_ANALYSIS = True  # If True, also run analysis using global PCs from all data

def get_final_token_activations(model, tokenizer, prompts, layer_idx, system_prompt=""):
    activations = []

    def hook_fn(module, input, output):
        last_token_activation = output[0][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    print(f"Extracting activations from layer {layer_idx}...")
    for prompt in tqdm(prompts, desc="Extracting activations"):
        if USE_SYSTEM_PROMPT_FOR_MANIFOLD and system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{'role': 'user', 'content': prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(DEVICE)
        with torch.no_grad():
            model(input_ids=inputs)

    hook_handle.remove()
    return torch.cat(activations, dim=0)

def get_global_activations(model, tokenizer, concept_prompts, layer_idx, system_prompt=""):
    """
    Extract activations from all prompts across all concepts to compute global PCs.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        concept_prompts: Dictionary mapping concept names to lists of prompts
        layer_idx: Layer index to extract activations from
        system_prompt: System prompt to use (if any)
    
    Returns:
        Tuple of (all_activations_tensor, prompt_to_concept_mapping, all_prompts_list)
    """
    all_prompts = []
    prompt_to_concept = []
    
    # Collect all prompts from all concepts
    for concept, prompts in concept_prompts.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_concept.append(concept)
    
    print(f"Extracting global activations from {len(all_prompts)} prompts across {len(concept_prompts)} concepts...")
    
    # Extract activations for all prompts
    global_activations = get_final_token_activations(
        model, tokenizer, all_prompts, layer_idx, system_prompt
    )
    
    return global_activations, prompt_to_concept, all_prompts

def get_enhanced_global_activations(model, tokenizer, concept_prompts, layer_idx, system_prompt=""):
    """
    Extract activations from all prompts across all concepts AND STSB dataset to compute global PCs.
    This creates a much larger and more diverse dataset for global PC computation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        concept_prompts: Dictionary mapping concept names to lists of prompts
        layer_idx: Layer index to extract activations from
        system_prompt: System prompt to use (if any)
    
    Returns:
        Tuple of (all_activations_tensor, prompt_to_source_mapping, all_prompts_list)
    """
    all_prompts = []
    prompt_to_source = []
    
    # Collect all prompts from all concepts in prompts.json
    concept_count = 0
    for concept, prompts in concept_prompts.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_source.append(f"concept_{concept}")
            concept_count += 1
    
    print(f"Loaded {concept_count} prompts from {len(concept_prompts)} concepts in prompts.json")
    
    # Load and add STSB dataset sentences
    try:
        from datasets import load_dataset
        print("Loading STSB dataset...")
        
        # Load the STSB dataset
        stsb_dataset = load_dataset("sentence-transformers/stsb", split="train")
        stsb_sentences = stsb_dataset["sentence2"]
        
        # Add STSB sentences to our dataset
        stsb_count = 0
        for sentence in stsb_sentences:
            if sentence and len(sentence.strip()) > 0:  # Skip empty sentences
                all_prompts.append(sentence.strip())
                prompt_to_source.append("stsb_dataset")
                stsb_count += 1
        
        print(f"Loaded {stsb_count} sentences from STSB dataset (sentence2 column, train split)")
        
    except ImportError:
        print("WARNING: 'datasets' library not available. Install with: pip install datasets")
        print("Continuing with only prompts.json data...")
    except Exception as e:
        print(f"WARNING: Failed to load STSB dataset: {e}")
        print("Continuing with only prompts.json data...")
    
    total_prompts = len(all_prompts)
    print(f"Total dataset size: {total_prompts} sentences ({concept_count} from prompts.json + {total_prompts - concept_count} from STSB)")
    print(f"Extracting global activations from this enhanced dataset at layer {layer_idx}...")
    
    # Extract activations for all prompts
    global_activations = get_final_token_activations(
        model, tokenizer, all_prompts, layer_idx, system_prompt
    )
    
    return global_activations, prompt_to_source, all_prompts

def analyse_global_manifolds(global_activations):
    """
    Compute global PCs from all activations combined.
    
    Args:
        global_activations: Tensor of shape (n_prompts, hidden_dim) containing all activations
    
    Returns:
        Dictionary containing global analysis results similar to analyse_manifolds output
    """
    print(f"Computing global PCs from {global_activations.shape[0]} activations...")
    print(f"Tensor device: {global_activations.device}, dtype: {global_activations.dtype}")
    
    # Store original dtype and device for later compatibility
    original_dtype = global_activations.dtype
    original_device = global_activations.device
    
    # Convert to float32 if in half precision (required for SVD)
    if global_activations.dtype == torch.float16:
        print("Converting from float16 to float32 for SVD computation...")
        global_activations = global_activations.float()
    
    # Keep on original device (likely GPU) for better performance
    device = global_activations.device
    
    # Center the activations
    mean_activation = global_activations.mean(dim=0)
    centered_activations = global_activations - mean_activation
    
    # Compute SVD to get principal components
    try:
        U, S, Vt = torch.svd(centered_activations)
        print(f"SVD completed successfully on {device}")
    except RuntimeError as e:
        print(f"SVD failed on {device}: {e}")
        if device.type == 'cuda':
            print("Trying SVD on CPU as fallback...")
            centered_activations_cpu = centered_activations.cpu()
            try:
                U, S, Vt = torch.svd(centered_activations_cpu)
                # Move results back to original device
                U, S, Vt = U.to(device), S.to(device), Vt.to(device)
                centered_activations = centered_activations_cpu.to(device)
                print("SVD completed successfully on CPU, results moved back to GPU")
            except RuntimeError as e2:
                print(f"SVD also failed on CPU: {e2}")
                print("Trying with double precision...")
                centered_activations_double = centered_activations_cpu.double()
                U, S, Vt = torch.svd(centered_activations_double)
                # Convert back to float32 and move to original device
                U, S, Vt = U.float().to(device), S.float().to(device), Vt.float().to(device)
                centered_activations = centered_activations_double.float().to(device)
                print("SVD completed with double precision, results converted back to float32")
        else:
            raise e
    
    # Extract eigenvalues and eigenvectors
    eigenvalues = S ** 2 / (global_activations.shape[0] - 1)
    eigenvectors = Vt.T  # Each column is an eigenvector
    
    # Keep results on the same device as input but ensure compatibility with helpers.py
    # Store in float32 for internal use, will convert when needed for helpers.py
    global_analysis = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors.T,  # Each row is an eigenvector (consistent with analyse_manifolds)
        "centered_acts": centered_activations,
        "mean": mean_activation,
        # Add metadata for compatibility
        "_original_dtype": original_dtype,
        "_original_device": original_device
    }
    
    print(f"Global PCA completed on {device}. Top 5 eigenvalues: {eigenvalues[:5].tolist()}")
    
    return global_analysis

def find_top_prompts_global(all_prompts, global_centered_acts, pc_direction, n=10, use_normalized_projection=True, prompt_sources=None):
    """
    Find top prompts aligned with a global PC direction.
    
    Args:
        all_prompts: List of all prompts
        global_centered_acts: Centered activations for all prompts
        pc_direction: Principal component direction vector
        n: Number of top prompts to return
        use_normalized_projection: Whether to normalize projections by vector magnitude
        prompt_sources: Optional list of source labels for each prompt
    
    Returns:
        Dictionary with 'positive' and 'negative' lists of top prompts (and sources if provided)
    """
    # Ensure tensors are on the same device and dtype
    target_device = global_centered_acts.device
    target_dtype = torch.float32  # Use float32 for numerical stability
    
    # Convert both tensors to same device and dtype
    global_centered_acts = global_centered_acts.to(device=target_device, dtype=target_dtype)
    pc_direction = pc_direction.to(device=target_device, dtype=target_dtype)
    
    if use_normalized_projection:
        # Normalize projections by vector magnitude
        vector_magnitudes = torch.norm(global_centered_acts, dim=1)
        # Add small epsilon to avoid division by zero
        projections = torch.matmul(global_centered_acts, pc_direction) / (vector_magnitudes + 1e-8)
    else:
        projections = torch.matmul(global_centered_acts, pc_direction)
    
    # Get indices for top positive and negative projections
    sorted_indices = torch.argsort(projections)
    
    # Top negative (most negative projections)
    top_negative_indices = sorted_indices[:n]
    # Top positive (most positive projections) 
    top_positive_indices = sorted_indices[-n:]
    top_positive_indices = torch.flip(top_positive_indices, dims=[0])  # Reverse to get highest first
    
    if prompt_sources is not None:
        top_prompts = {
            'positive': [(all_prompts[i.item()], prompt_sources[i.item()]) for i in top_positive_indices],
            'negative': [(all_prompts[i.item()], prompt_sources[i.item()]) for i in top_negative_indices]
        }
    else:
        top_prompts = {
            'positive': [all_prompts[i.item()] for i in top_positive_indices],
            'negative': [all_prompts[i.item()] for i in top_negative_indices]
        }
    
    return top_prompts

def compute_pc_cosine_similarity(concept_analysis, global_analysis, concept_name, layer_idx, top_k=5):
    """
    Compute and display cosine similarity matrix between top concept-specific and global PCs.
    
    Args:
        concept_analysis: Analysis results for the specific concept
        global_analysis: Global analysis results 
        concept_name: Name of the concept being analyzed
        layer_idx: Layer index
        top_k: Number of top PCs to compare (default: 5)
    """
    # Extract top k eigenvectors (already sorted by eigenvalue in descending order)
    concept_pcs = concept_analysis["eigenvectors"][:top_k]  # Shape: (top_k, hidden_dim)
    global_pcs = global_analysis["eigenvectors"][:top_k]    # Shape: (top_k, hidden_dim)
    
    print(f"Before alignment - Concept PCs: device={concept_pcs.device}, dtype={concept_pcs.dtype}")
    print(f"Before alignment - Global PCs: device={global_pcs.device}, dtype={global_pcs.dtype}")
    
    # Choose target device (prefer GPU if available)
    target_device = global_pcs.device if global_pcs.device.type == 'cuda' else concept_pcs.device
    
    # Choose target dtype (use float32 for numerical stability)
    target_dtype = torch.float32
    
    # Ensure both tensors are on same device and dtype
    concept_pcs = concept_pcs.to(device=target_device, dtype=target_dtype)
    global_pcs = global_pcs.to(device=target_device, dtype=target_dtype)
    
    print(f"After alignment - Computing cosine similarity on {target_device} with dtype {target_dtype}")
    
    # Ensure tensors have the same shape in the hidden dimension
    min_hidden_dim = min(concept_pcs.shape[1], global_pcs.shape[1])
    if concept_pcs.shape[1] != global_pcs.shape[1]:
        print(f"Warning: Dimension mismatch - concept PCs: {concept_pcs.shape[1]}, global PCs: {global_pcs.shape[1]}")
        print(f"Truncating to minimum dimension: {min_hidden_dim}")
        concept_pcs = concept_pcs[:, :min_hidden_dim]
        global_pcs = global_pcs[:, :min_hidden_dim]
    
    # Ensure we don't have more PCs than available
    actual_top_k = min(top_k, concept_pcs.shape[0], global_pcs.shape[0])
    if actual_top_k < top_k:
        print(f"Warning: Requested {top_k} PCs but only {actual_top_k} available. Using {actual_top_k}.")
        concept_pcs = concept_pcs[:actual_top_k]
        global_pcs = global_pcs[:actual_top_k]
        top_k = actual_top_k
    
    # Compute cosine similarity matrix
    # Normalize vectors for cosine similarity (add epsilon to avoid division by zero)
    concept_norms = torch.norm(concept_pcs, dim=1, keepdim=True)
    global_norms = torch.norm(global_pcs, dim=1, keepdim=True)
    
    concept_pcs_norm = concept_pcs / (concept_norms + 1e-8)
    global_pcs_norm = global_pcs / (global_norms + 1e-8)
    
    # Cosine similarity matrix: concept_pcs @ global_pcs.T
    cosine_sim_matrix = torch.matmul(concept_pcs_norm, global_pcs_norm.T)
    
    print("\n" + "*"*80)
    print(f"### COSINE SIMILARITY: {concept_name.upper()} PCs vs GLOBAL PCs (Layer {layer_idx}) ###")
    print("*"*80)
    print(f"Matrix: {concept_name} PCs (rows) vs Global PCs (columns)")
    print(f"Each PC ordered by eigenvalue (largest first)")
    print()
    
    # Print header
    header = f"{'':>12}"
    for j in range(top_k):
        header += f"{'Global-PC' + str(j):>12}"
    print(header)
    print("-" * (12 + 12 * top_k))
    
    # Print matrix rows
    for i in range(top_k):
        row = f"{concept_name + '-PC' + str(i):>12}"
        for j in range(top_k):
            similarity = cosine_sim_matrix[i, j].item()
            row += f"{similarity:>12.4f}"
        print(row)
    
    print()
    
    # Find and report highest similarities
    max_similarity = torch.max(torch.abs(cosine_sim_matrix)).item()
    max_pos = torch.unravel_index(torch.argmax(torch.abs(cosine_sim_matrix)), cosine_sim_matrix.shape)
    max_i, max_j = max_pos[0].item(), max_pos[1].item()
    actual_similarity = cosine_sim_matrix[max_i, max_j].item()
    
    print(f"Highest absolute similarity: {abs(actual_similarity):.4f}")
    print(f"Between {concept_name}-PC{max_i} and Global-PC{max_j} (similarity = {actual_similarity:.4f})")
    
    # Report average similarities
    avg_similarity = torch.mean(torch.abs(cosine_sim_matrix)).item()
    print(f"Average absolute similarity: {avg_similarity:.4f}")
    
    print("*"*80 + "\n")
    
    return cosine_sim_matrix

def main():
    print(f"\nConfiguration: PERTURB_ONCE is set to {PERTURB_ONCE}\n")
    print(f"Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD is set to {USE_SYSTEM_PROMPT_FOR_MANIFOLD}\n")
    print(f"Configuration: USE_NORMALIZED_PROJECTION is set to {USE_NORMALIZED_PROJECTION}\n")
    print(f"Configuration: RUN_GLOBAL_PC_ANALYSIS is set to {RUN_GLOBAL_PC_ANALYSIS}\n")
    if RUN_GLOBAL_PC_ANALYSIS:
        print("Note: Global PC analysis uses enhanced dataset including:")
        print("  - All sentences from prompts.json")
        print("  - All sentences from STSB dataset (sentence2 column, train split: ~5.75k sentences)")
        print("  - Requires 'datasets' library: pip install datasets\n")

    dog_avg_eigenvalues = {}
    dog_top_eigenvectors = {}
    global_analysis_cache = {}  # Cache global analysis results by layer
    model_name_str = MODEL_NAME.split('/')[-1]

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    with open('prompts.json', 'r', encoding='utf-8') as f:
        concept_prompts = json.load(f)
    
    TARGET_LAYERS = [0, 15, 31] # Llama-3.1-8B has 32 layers (0-31)
    AXES_TO_ANALYZE = range(5)

    # Define concept-prompt pairings
    concept_prompt_pairs = [
        {
            "concept": "lion",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Please write a sentence about lions."
        },
        {
            "concept": "dog",
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "Please write a sentence about dogs."
        }
    ]

    # For each concept-prompt pairing, run through all layers
    for pair_idx, pair in enumerate(concept_prompt_pairs):
        concept = pair["concept"]
        system_prompt = pair["system_prompt"]
        user_prompt = pair["user_prompt"]
        
        print("\n" + "#"*100)
        print(f"### STARTING ANALYSIS FOR CONCEPT-PROMPT PAIR {pair_idx + 1}/{len(concept_prompt_pairs)} ###")
        print(f"### CONCEPT: '{concept}' ###")
        print(f"### SYSTEM PROMPT: '{system_prompt}' ###")
        print(f"### USER PROMPT: '{user_prompt}' ###")
        print("#"*100 + "\n")
        
        # Check if concept exists in the dataset
        if concept not in concept_prompts:
            print(f"Concept '{concept}' not found in prompts.json. Available concepts: {list(concept_prompts.keys())}")
            print("Skipping this concept-prompt pair.\n")
            continue

        for target_layer in TARGET_LAYERS:
            print("\n" + "="*80)
            print(f"### ANALYZING LAYER {target_layer} for concept '{concept}' ###")
            print("="*80 + "\n")

            # Extract activations for this specific concept at this layer
            system_prompt_for_manifold = "You are a language model assistant. Please translate the following text accurately from English into German:" if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
            
            concept_activations = get_final_token_activations(
                model, tokenizer, concept_prompts[concept], target_layer, 
                system_prompt=system_prompt_for_manifold
            )
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Analyze manifold for this concept
            all_activations = {concept: concept_activations}
            analysis_results = analyse_manifolds(all_activations)
            
            if concept not in analysis_results:
                print(f"Analysis for concept '{concept}' failed for layer {target_layer}. Skipping to next layer.")
                continue

            # Store dog analysis for overall plotting (if concept is dog)
            if concept == "dog":
                dog_analysis = analysis_results[concept]
                dog_avg_eigenvalues[target_layer] = dog_analysis["eigenvalues"].mean().item()
                dog_top_eigenvectors[target_layer] = dog_analysis["eigenvectors"][0]

            # GLOBAL PC ANALYSIS - Extract and compute global PCs if not already cached
            if RUN_GLOBAL_PC_ANALYSIS and target_layer not in global_analysis_cache:
                print("\n" + "~"*80)
                print(f"### COMPUTING ENHANCED GLOBAL PCs FOR LAYER {target_layer} ###")
                print(f"### (prompts.json + STSB dataset for broader representation) ###")
                print("~"*80 + "\n")
                
                try:
                    global_activations, prompt_to_source, all_prompts = get_enhanced_global_activations(
                        model, tokenizer, concept_prompts, target_layer, 
                        system_prompt=system_prompt_for_manifold
                    )
                    
                    global_analysis = analyse_global_manifolds(global_activations)
                    global_analysis_cache[target_layer] = {
                        'analysis': global_analysis,
                        'all_prompts': all_prompts,
                        'prompt_to_source': prompt_to_source
                    }
                    
                    # Memory cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"Global PC analysis cached for layer {target_layer}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to compute global PC analysis for layer {target_layer}: {e}")
                    print("Skipping global PC analysis for this layer.")
                    import traceback
                    traceback.print_exc()
                    continue
                
                print("~"*80 + "\n")

            # COSINE SIMILARITY ANALYSIS - Compare concept PCs with global PCs
            if RUN_GLOBAL_PC_ANALYSIS and target_layer in global_analysis_cache:
                global_data = global_analysis_cache[target_layer]
                global_analysis = global_data['analysis']
                
                try:
                    # Compute and display cosine similarity matrix
                    cosine_sim_matrix = compute_pc_cosine_similarity(
                        analysis_results[concept], 
                        global_analysis, 
                        concept, 
                        target_layer, 
                        top_k=5
                    )
                except Exception as e:
                    print(f"ERROR: Failed to compute cosine similarity for {concept} at layer {target_layer}: {e}")
                    print("Skipping cosine similarity analysis for this concept-layer combination.")
                    import traceback
                    traceback.print_exc()

            print(f"\nRunning experiments for:")
            print(f"System prompt: '{system_prompt}'")
            print(f"User prompt: '{user_prompt}'")
            print(f"Concept: '{concept}'")
            print(f"Layer: {target_layer}")

            messages_to_test = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            run_perturbation_experiment(
                model, tokenizer, messages_to_test, target_layer, 
                analysis_results[concept], concept, AXES_TO_ANALYZE, 
                target_token_idx=None, perturb_once=PERTURB_ONCE, 
                orthogonal_mode=False, use_largest_eigenvalue=True  # actual orthogonal eigenval too small
            )
            
            # Display top prompts aligned with the PC direction
            for axis in AXES_TO_ANALYZE:
                if axis >= len(analysis_results[concept]["eigenvectors"]):
                    break
                    
                pc_direction = analysis_results[concept]["eigenvectors"][axis]
                
                print("\n" + "="*80)
                print(f"--- Analyzing original dataset prompts along the PC{axis} '{concept}' direction (Layer {target_layer}) ---")
                if USE_NORMALIZED_PROJECTION:
                    print("Using normalized projections (projection magnitude / vector magnitude)")
                else:
                    print("Using raw projections")
                top_prompts = find_top_prompts(
                    concept_prompts[concept],
                    analysis_results[concept]["centered_acts"],
                    pc_direction,
                    n=10,
                    use_normalized_projection=USE_NORMALIZED_PROJECTION
                )

                print(f"\nTop 10 prompts most aligned with POSITIVE PC{axis} direction:")
                for i, prompt in enumerate(top_prompts['positive'], 1):
                    print(f"{i:2d}. '{prompt}'")
                    
                print(f"\nTop 10 prompts most aligned with NEGATIVE PC{axis} direction:")
                for i, prompt in enumerate(top_prompts['negative'], 1):
                    print(f"{i:2d}. '{prompt}'")
                print("="*80)
            
            # --- PROJECTION-BASED PERTURBATION (COMMENTED OUT) ---
            # For each PC, run a projection-based perturbation that zeroes out all other PCs
            # for axis in AXES_TO_ANALYZE:
            #     if axis >= len(analysis_results[concept]["eigenvectors"]):
            #         break
            #         
            #     # Run projection-based perturbation for this PC
            #     run_projection_based_perturbation(
            #         model, tokenizer, messages_to_test, target_layer,
            #         analysis_results[concept], concept, axis,
            #         target_token_idx=None, perturb_once=PERTURB_ONCE
            #     )

            # --- ORTHOGONAL PERTURBATION ---
            run_perturbation_experiment(
                model, tokenizer, messages_to_test, target_layer,
                analysis_results[concept], concept,
                target_token_idx=None, perturb_once=PERTURB_ONCE,
                orthogonal_mode=True, use_largest_eigenvalue=True
            )

            # --- ABLATION EXPERIMENT ---
            run_ablation_experiment(
                model, tokenizer, messages_to_test, target_layer,
                analysis_results[concept], concept,
                target_token_idx=None, perturb_once=PERTURB_ONCE
            )

            # GLOBAL PC ANALYSIS - Run perturbation experiments using global PCs
            if RUN_GLOBAL_PC_ANALYSIS and target_layer in global_analysis_cache:
                global_data = global_analysis_cache[target_layer]
                global_analysis = global_data['analysis']
                all_prompts = global_data['all_prompts']
                
                # Ensure global analysis is compatible with helpers.py functions (CPU, appropriate dtype)
                global_analysis_for_helpers = {}
                for key, tensor in global_analysis.items():
                    if torch.is_tensor(tensor):
                        # Convert to CPU and float16 to match helpers.py expectations
                        global_analysis_for_helpers[key] = tensor.cpu().to(dtype=torch.float16)
                    else:
                        global_analysis_for_helpers[key] = tensor
                
                print("\n" + "@"*80)
                print(f"### GLOBAL PC PERTURBATION EXPERIMENTS FOR LAYER {target_layer} ###")
                print(f"### Using Global PCs computed from {len(all_prompts)} sentences ###")
                print(f"### (prompts.json + STSB dataset for enhanced global representation) ###")
                print("@"*80 + "\n")
                
                # Run perturbation experiment using global PCs
                print(f"Running GLOBAL PC perturbation experiments for concept '{concept}'...")
                try:
                    run_perturbation_experiment(
                        model, tokenizer, messages_to_test, target_layer, 
                        global_analysis_for_helpers, f"GLOBAL-{concept}", AXES_TO_ANALYZE, 
                        target_token_idx=None, perturb_once=PERTURB_ONCE, 
                        orthogonal_mode=False, use_largest_eigenvalue=True
                    )
                except Exception as e:
                    print(f"ERROR in global PC perturbation experiment: {e}")
                    print("Continuing with other analyses...")
                
                # Display top prompts aligned with the global PC directions
                for axis in AXES_TO_ANALYZE:
                    if axis >= len(global_analysis["eigenvectors"]):
                        break
                        
                    global_pc_direction = global_analysis["eigenvectors"][axis]
                    
                    print("\n" + "="*80)
                    print(f"--- Analyzing ALL dataset prompts along the GLOBAL PC{axis} direction (Layer {target_layer}) ---")
                    if USE_NORMALIZED_PROJECTION:
                        print("Using normalized projections (projection magnitude / vector magnitude)")
                    else:
                        print("Using raw projections")
                    
                    # Get prompt sources from cached data
                    prompt_sources = global_data.get('prompt_to_source', None)
                    
                    try:
                        top_prompts_global = find_top_prompts_global(
                            all_prompts,
                            global_analysis["centered_acts"],
                            global_pc_direction,
                            n=10,
                            use_normalized_projection=USE_NORMALIZED_PROJECTION,
                            prompt_sources=prompt_sources
                        )

                        print(f"\nTop 10 sentences most aligned with POSITIVE GLOBAL PC{axis} direction:")
                        for i, item in enumerate(top_prompts_global['positive'], 1):
                            if isinstance(item, tuple):
                                prompt, source = item
                                source_display = source.replace('concept_', '').replace('stsb_dataset', 'STSB')
                                print(f"{i:2d}. [{source_display}] '{prompt}'")
                            else:
                                print(f"{i:2d}. '{item}'")
                            
                        print(f"\nTop 10 sentences most aligned with NEGATIVE GLOBAL PC{axis} direction:")
                        for i, item in enumerate(top_prompts_global['negative'], 1):
                            if isinstance(item, tuple):
                                prompt, source = item
                                source_display = source.replace('concept_', '').replace('stsb_dataset', 'STSB')
                                print(f"{i:2d}. [{source_display}] '{prompt}'")
                            else:
                                print(f"{i:2d}. '{item}'")
                                
                    except Exception as e:
                        print(f"ERROR in global PC prompt analysis for axis {axis}: {e}")
                        print("Continuing with next axis...")
                        
                    print("="*80)
                
                # --- GLOBAL PC ORTHOGONAL PERTURBATION ---
                print(f"\nRunning GLOBAL PC orthogonal perturbation for concept '{concept}'...")
                try:
                    run_perturbation_experiment(
                        model, tokenizer, messages_to_test, target_layer,
                        global_analysis_for_helpers, f"GLOBAL-{concept}",
                        target_token_idx=None, perturb_once=PERTURB_ONCE,
                        orthogonal_mode=True, use_largest_eigenvalue=True
                    )
                except Exception as e:
                    print(f"ERROR in global PC orthogonal perturbation: {e}")
                    print("Continuing with other analyses...")

                # --- GLOBAL PC ABLATION EXPERIMENT ---
                print(f"\nRunning GLOBAL PC ablation experiment for concept '{concept}'...")
                try:
                    run_ablation_experiment(
                        model, tokenizer, messages_to_test, target_layer,
                        global_analysis_for_helpers, f"GLOBAL-{concept}",
                        target_token_idx=None, perturb_once=PERTURB_ONCE
                    )
                except Exception as e:
                    print(f"ERROR in global PC ablation experiment: {e}")
                    print("Continuing with other analyses...")
                
                print("@"*80 + "\n")

    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "lastToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "lastToken")

if __name__ == "__main__":
    main()