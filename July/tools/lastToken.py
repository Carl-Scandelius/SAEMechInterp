"""Transformer concept manifold exploration and perturbation analysis."""

from __future__ import annotations

import torch
from tqdm import tqdm
import gc
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence
from helpers import (
    get_model_and_tokenizer,
    analyse_manifolds,
    find_top_prompts,
    plot_avg_eigenvalues,
    plot_similarity_matrix,
    run_perturbation_experiment,
    run_ablation_experiment,
    generate_with_perturbation,
    MODEL_NAME,
    DEVICE,
)
from transformers import logging
logging.set_verbosity(40)

USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = False
USE_NORMALIZED_PROJECTION = True
RUN_GLOBAL_PC_ANALYSIS = True
CROSS_CONCEPT_ONLY = False
USE_PRANAV_SENTENCES = False
LOCAL_CENTRE = False

def get_final_token_activations(
    model, tokenizer, prompts, layer_idx, system_prompt=""
):
    """Extract final token activations from specified layer."""
    activations = []

    def hook_fn(module, input, output):
        last_token_activation = output[0][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        print(f"Extracting activations from layer {layer_idx}...")
        for prompt in tqdm(prompts, desc="Extracting activations"):
            if USE_SYSTEM_PROMPT_FOR_MANIFOLD and system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{'role': 'user', 'content': prompt}]

            try:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=False
                ).to(DEVICE)
                
                with torch.no_grad():
                    model(input_ids=inputs)
            except Exception as e:
                print(f"Warning: Failed to process prompt '{prompt[:50]}...': {e}")
                continue
    finally:
        # Ensure hook is always removed
        hook_handle.remove()
        
    if not activations:
        print(f"Warning: No activations extracted from layer {layer_idx}")
        return torch.empty(0, model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096, device='cpu')
        
    return torch.cat(activations, dim=0)

def get_global_activations(
    model, tokenizer, concept_prompts, layer_idx, system_prompt=""
):
    """Extract activations from all prompts across concepts for global PC computation."""
    all_prompts = []
    prompt_to_concept = []
    
    for concept, prompts in concept_prompts.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_concept.append(concept)
    
    print(f"Extracting global activations from {len(all_prompts)} prompts across {len(concept_prompts)} concepts...")
    
    global_activations = get_final_token_activations(
        model, tokenizer, all_prompts, layer_idx, system_prompt
    )
    
    return global_activations, prompt_to_concept, all_prompts

def get_enhanced_global_activations(
    model, tokenizer, concept_prompts, layer_idx, system_prompt=""
):
    """Extract activations from all prompts plus STSB dataset for enhanced global PC computation."""
    all_prompts = []
    prompt_to_source = []
    
    concept_count = 0
    for concept, prompts in concept_prompts.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_source.append(f"concept_{concept}")
            concept_count += 1
    
    print(f"Loaded {concept_count} prompts from {len(concept_prompts)} concepts in prompts.json")
    
    try:
        from datasets import load_dataset
        print("Loading STSB dataset...")
        
        stsb_dataset = load_dataset("sentence-transformers/stsb", split="train")
        stsb_sentences = stsb_dataset["sentence2"]
        
        stsb_count = 0
        for sentence in stsb_sentences:
            if sentence and len(sentence.strip()) > 0:
                all_prompts.append(sentence.strip())
                prompt_to_source.append("stsb_dataset")
                stsb_count += 1
        
        print(f"Loaded {stsb_count} sentences from STSB dataset")
        
    except ImportError:
        print("WARNING: 'datasets' library not available. Install with: pip install datasets")
        print("Continuing with only prompts.json data...")
    except Exception as e:
        print(f"WARNING: Failed to load STSB dataset: {e}")
        print("Continuing with only prompts.json data...")
    
    total_prompts = len(all_prompts)
    print(f"Total dataset: {total_prompts} sentences ({concept_count} from prompts.json + {total_prompts - concept_count} from STSB)")
    print(f"Extracting global activations at layer {layer_idx}...")
    
    global_activations = get_final_token_activations(
        model, tokenizer, all_prompts, layer_idx, system_prompt
    )
    
    return global_activations, prompt_to_source, all_prompts

def analyse_global_manifolds(global_activations):
    """Compute global PCs via SVD from combined activations."""
    print(f"Computing global PCs from {global_activations.shape[0]} activations...")
    print(f"Tensor device: {global_activations.device}, dtype: {global_activations.dtype}")
    
    original_dtype = global_activations.dtype
    original_device = global_activations.device
    
    # Move to CPU and use float32 for numerical stability
    print("Moving to CPU and converting to float32 for stable SVD computation...")
    global_activations = global_activations.cpu().float()
    
    mean_activation = global_activations.mean(dim=0)
    centered_activations = global_activations - mean_activation
    
    try:
        U, S, Vt = torch.svd(centered_activations)
        print(f"SVD completed successfully on CPU")
    except RuntimeError as e:
        print(f"SVD failed on CPU: {e}")
        print("Trying with double precision...")
        try:
            centered_activations_double = centered_activations.double()
            U, S, Vt = torch.svd(centered_activations_double)
            U, S, Vt = U.float(), S.float(), Vt.float()
            centered_activations = centered_activations_double.float()
            print("SVD completed with double precision")
        except RuntimeError as e2:
            print(f"SVD also failed with double precision: {e2}")
            raise RuntimeError(f"SVD computation failed: {e2}")
    
    eigenvalues = S ** 2 / (global_activations.shape[0] - 1)
    eigenvectors = Vt.T
    
    # Move results back to original device if needed, but keep float32
    target_device = original_device if original_device.type == 'cuda' else torch.device('cpu')
    
    global_analysis = {
        "eigenvalues": eigenvalues.to(target_device),
        "eigenvectors": eigenvectors.T.to(target_device),  # Each row is an eigenvector
        "centered_acts": centered_activations.to(target_device),
        "mean": mean_activation.to(target_device),
        "_original_dtype": original_dtype,
        "_original_device": original_device
    }
    
    print(f"Global PCA completed. Results moved to {target_device}. Top 5 eigenvalues: {eigenvalues[:5].tolist()}")
    
    return global_analysis

def find_top_prompts_global(
    all_prompts, 
    global_centered_acts, 
    pc_direction, 
    n=10, 
    use_normalized_projection=True, 
    prompt_sources=None
):
    """Find top prompts aligned with global PC direction."""
    target_device = global_centered_acts.device
    target_dtype = torch.float32
    
    global_centered_acts = global_centered_acts.to(device=target_device, dtype=target_dtype)
    pc_direction = pc_direction.to(device=target_device, dtype=target_dtype)
    
    if use_normalized_projection:
        vector_magnitudes = torch.norm(global_centered_acts, dim=1)
        projections = torch.matmul(global_centered_acts, pc_direction) / (vector_magnitudes + 1e-8)
    else:
        projections = torch.matmul(global_centered_acts, pc_direction)
    
    sorted_indices = torch.argsort(projections)
    
    top_negative_indices = sorted_indices[:n]
    top_positive_indices = sorted_indices[-n:]
    top_positive_indices = torch.flip(top_positive_indices, dims=[0])
    
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

def compute_pc_cosine_similarity(
    concept_analysis, 
    global_analysis, 
    concept_name, 
    layer_idx, 
    top_k=5
):
    """Compute cosine similarity matrix between concept-specific and global PCs."""
    concept_pcs = concept_analysis["eigenvectors"][:top_k]
    global_pcs = global_analysis["eigenvectors"][:top_k]
    
    print(f"Before alignment - Concept PCs: device={concept_pcs.device}, dtype={concept_pcs.dtype}")
    print(f"Before alignment - Global PCs: device={global_pcs.device}, dtype={global_pcs.dtype}")
    
    target_device = global_pcs.device if global_pcs.device.type == 'cuda' else concept_pcs.device
    target_dtype = torch.float32
    
    concept_pcs = concept_pcs.to(device=target_device, dtype=target_dtype)
    global_pcs = global_pcs.to(device=target_device, dtype=target_dtype)
    
    print(f"After alignment - Computing cosine similarity on {target_device} with dtype {target_dtype}")
    
    min_hidden_dim = min(concept_pcs.shape[1], global_pcs.shape[1])
    if concept_pcs.shape[1] != global_pcs.shape[1]:
        print(f"Warning: Dimension mismatch - concept PCs: {concept_pcs.shape[1]}, global PCs: {global_pcs.shape[1]}")
        print(f"Truncating to minimum dimension: {min_hidden_dim}")
        concept_pcs = concept_pcs[:, :min_hidden_dim]
        global_pcs = global_pcs[:, :min_hidden_dim]
    
    actual_top_k = min(top_k, concept_pcs.shape[0], global_pcs.shape[0])
    if actual_top_k < top_k:
        print(f"Warning: Requested {top_k} PCs but only {actual_top_k} available. Using {actual_top_k}.")
        concept_pcs = concept_pcs[:actual_top_k]
        global_pcs = global_pcs[:actual_top_k]
        top_k = actual_top_k
    
    concept_norms = torch.norm(concept_pcs, dim=1, keepdim=True)
    global_norms = torch.norm(global_pcs, dim=1, keepdim=True)
    
    concept_pcs_norm = concept_pcs / (concept_norms + 1e-8)
    global_pcs_norm = global_pcs / (global_norms + 1e-8)
    
    cosine_sim_matrix = torch.matmul(concept_pcs_norm, global_pcs_norm.T)
    
    print("\n" + "*"*80)
    print(f"### COSINE SIMILARITY: {concept_name.upper()} PCs vs GLOBAL PCs (Layer {layer_idx}) ###")
    print("*"*80)
    print(f"Matrix: {concept_name} PCs (rows) vs Global PCs (columns)")
    print(f"Each PC ordered by eigenvalue (largest first)")
    print()
    
    header = f"{'':>12}"
    for j in range(top_k):
        header += f"{'Global-PC' + str(j):>12}"
    print(header)
    print("-" * (12 + 12 * top_k))
    
    for i in range(top_k):
        row = f"{concept_name + '-PC' + str(i):>12}"
        for j in range(top_k):
            similarity = cosine_sim_matrix[i, j].item()
            row += f"{similarity:>12.4f}"
        print(row)
    
    print()
    
    max_similarity = torch.max(torch.abs(cosine_sim_matrix)).item()
    max_pos = torch.unravel_index(torch.argmax(torch.abs(cosine_sim_matrix)), cosine_sim_matrix.shape)
    max_i, max_j = max_pos[0].item(), max_pos[1].item()
    actual_similarity = cosine_sim_matrix[max_i, max_j].item()
    
    print(f"Highest absolute similarity: {abs(actual_similarity):.4f}")
    print(f"Between {concept_name}-PC{max_i} and Global-PC{max_j} (similarity = {actual_similarity:.4f})")
    
    avg_similarity = torch.mean(torch.abs(cosine_sim_matrix)).item()
    print(f"Average absolute similarity: {avg_similarity:.4f}")
    
    print("*"*80 + "\n")
    
    return cosine_sim_matrix

def run_cross_concept_perturbation(
    model, tokenizer, messages, layer_idx,
    source_concept_analysis, target_concept_analysis,
    source_concept_name, target_concept_name,
    target_token_idx=None, perturb_once=False
):
    """
    Perturb along the direction between two concept manifolds.
    """
    # Get centroids (mean activations) for both concepts
    source_centroid = source_concept_analysis["centroid"].to(DEVICE)
    target_centroid = target_concept_analysis["centroid"].to(DEVICE)
    
    # Compute the vector between centroids
    cross_concept_vector = target_centroid - source_centroid
    cross_concept_distance = torch.norm(cross_concept_vector).item()
    
    # Normalize the direction vector
    direction_vector = cross_concept_vector / torch.norm(cross_concept_vector)
    
    # Prepare inputs for generation
    if not isinstance(messages, torch.Tensor):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    else:
        inputs = messages
    
    system_prompt = messages[0]['content'] if messages[0]['role'] == 'system' else ""
    user_prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
    
    # Check centering mode
    centering_mode = source_concept_analysis.get("local_centre", False)
    centering_info = "LOCAL centering (concept-specific PCs)" if centering_mode else "GLOBAL centering (cross-concept comparable PCs)"
    
    print("\n" + "="*80)
    print(f"--- CROSS-CONCEPT PERTURBATION: '{source_concept_name}' → '{target_concept_name}' ---")
    print(f"--- LAYER: {layer_idx} ---")
    print(f"--- Cross-concept distance: {cross_concept_distance:.4f} ---")
    print(f"--- Centering mode: {centering_info} ---")
    print("="*80)
    
    print(f"\nSystem: '{system_prompt}'")
    print(f"User: '{user_prompt}'")
    
    # Generate original output (0% perturbation)
    original_output = generate_with_perturbation(
        model, tokenizer, inputs, layer_idx, direction_vector, 0.0, 
        torch.tensor(1.0, device=model.device), target_token_idx, perturb_once
    )
    print(f"Original (0%): {original_output}")
    
    # Perturb in steps of 0.2 from 0.2 to 2.0 (twice the vector magnitude)
    print(f"\n--- Perturbing along {source_concept_name} → {target_concept_name} direction ---")
    for step in range(1, 11):  # 1, 2, 3, ..., 10 corresponding to 0.2, 0.4, 0.6, ..., 2.0
        percentage = step * 0.2
        
        perturbation_magnitude = percentage * cross_concept_distance
        
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, layer_idx, direction_vector, 
            perturbation_magnitude, torch.tensor(1.0, device=model.device),
            target_token_idx, perturb_once
        )
        
        print(f"({percentage*100:.0f}% of mag): {perturbed_output}")
    
    # Project 100% centroid perturbation onto target manifold
    print(f"\n--- Projecting 100% {source_concept_name} → {target_concept_name} perturbation onto {target_concept_name} manifold ---")
    
    # Get the 100% perturbation vector
    full_perturbation_magnitude = cross_concept_distance
    
    # Project onto target concept manifold using only its EFFECTIVE PCs
    effective_mask = target_concept_analysis["effective_mask"]
    all_eigenvectors = target_concept_analysis["eigenvectors"].to(DEVICE)
    target_eigenvectors = all_eigenvectors[effective_mask]  # Only effective PCs
    target_centroid_device = target_concept_analysis["centroid"].to(DEVICE)
    
    num_effective_pcs = effective_mask.sum().item()
    print(f"Using {num_effective_pcs} effective PCs out of {len(all_eigenvectors)} total PCs for {target_concept_name} manifold projection")
    
    # Create projection function
    def project_onto_manifold_hook(perturbation_vector, target_eigenvectors, target_centroid):
        def projection_hook(module, input_tensor, output):
            hidden_states = output[0]
            token_idx = target_token_idx if target_token_idx is not None else -1
            
            # Apply the original perturbation
            perturbed_activation = hidden_states[0, token_idx] + perturbation_vector
            
            # Center relative to target manifold centroid
            centered_activation = perturbed_activation - target_centroid
            
            # Project onto target manifold subspace (only effective PCs)
            coefficients = torch.matmul(centered_activation, target_eigenvectors.T)
            projected_activation = target_centroid + torch.matmul(coefficients, target_eigenvectors)
            
            modified_hidden_states = hidden_states.clone()
            modified_hidden_states[0, token_idx] = projected_activation
            return (modified_hidden_states,) + output[1:]
        return projection_hook
    
    perturbation_vector = direction_vector * full_perturbation_magnitude
    hook_handle = model.model.layers[layer_idx].register_forward_hook(
        project_onto_manifold_hook(perturbation_vector, target_eigenvectors, target_centroid_device)
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs, max_new_tokens=50, do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    hook_handle.remove()
    prompt_length = inputs.shape[1]
    projected_output = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    print(f"Projected onto {target_concept_name} manifold: {projected_output}")
    
    # Centroid position vector perturbations
    print(f"\n--- Perturbing along centroid position vectors ---")
    
    # Source concept centroid position vector perturbation
    source_centroid_norm = torch.norm(source_centroid).item()
    source_position_direction = source_centroid / torch.norm(source_centroid)
    step_size_source = 0.5 * source_centroid_norm
    
    print(f"\n{source_concept_name} centroid distance from origin: {source_centroid_norm:.4f}")
    print(f"Step size (0.5 * distance): {step_size_source:.4f}")
    print(f"Perturbing along {source_concept_name} centroid position vector:")
    
    for step in range(1, 6):  # 5 steps: 0.5x, 1.0x, 1.5x, 2.0x, 2.5x
        magnitude = step * step_size_source
        
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, layer_idx, source_position_direction,
            magnitude, torch.tensor(1.0, device=model.device),
            target_token_idx, perturb_once
        )
        
        print(f"Step {step} ({step * 0.5:.1f}x): {perturbed_output}")
    
    # Target concept centroid position vector perturbation  
    target_centroid_norm = torch.norm(target_centroid).item()
    target_position_direction = target_centroid / torch.norm(target_centroid)
    step_size_target = 0.5 * target_centroid_norm
    
    print(f"\n{target_concept_name} centroid distance from origin: {target_centroid_norm:.4f}")
    print(f"Step size (0.5 * distance): {step_size_target:.4f}")
    print(f"Perturbing along {target_concept_name} centroid position vector:")
    
    for step in range(1, 6):  # 5 steps: 0.5x, 1.0x, 1.5x, 2.0x, 2.5x
        magnitude = step * step_size_target
        
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, layer_idx, target_position_direction,
            magnitude, torch.tensor(1.0, device=model.device),
            target_token_idx, perturb_once
        )
        
        print(f"Step {step} ({step * 0.5:.1f}x): {perturbed_output}")
    
    print("="*80)

def main():
    print(f"\nConfiguration: PERTURB_ONCE={PERTURB_ONCE}")
    print(f"Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD={USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
    print(f"Configuration: USE_NORMALIZED_PROJECTION={USE_NORMALIZED_PROJECTION}")
    print(f"Configuration: RUN_GLOBAL_PC_ANALYSIS={RUN_GLOBAL_PC_ANALYSIS}")
    print(f"Configuration: CROSS_CONCEPT_ONLY={CROSS_CONCEPT_ONLY}")
    print(f"Configuration: USE_PRANAV_SENTENCES={USE_PRANAV_SENTENCES}")
    print(f"Configuration: LOCAL_CENTRE={LOCAL_CENTRE}\n")

    dog_avg_eigenvalues = {}
    dog_top_eigenvectors = {}
    global_analysis_cache = {}
    # Store analysis results for cross-concept perturbation
    all_concept_analyses = {}
    model_name_str = MODEL_NAME.split('/')[-1]

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    # Load appropriate dataset based on flag
    if USE_PRANAV_SENTENCES:
        print("Loading manifold_sentences_hard_exactword_1000.json...")
        with open('manifold_sentences_hard_exactword_1000.json', 'r', encoding='utf-8') as f:
            concept_prompts = json.load(f)
        print(f"Loaded {len(concept_prompts)} concepts from manifold sentences dataset")
    else:
        print("Loading prompts.json...")
        with open('prompts.json', 'r', encoding='utf-8') as f:
            concept_prompts = json.load(f)
        print(f"Loaded {len(concept_prompts)} concepts from prompts dataset")
    
    TARGET_LAYERS = [0, 15, 31]

    # Set up concept-prompt pairs based on the dataset being used
    if USE_PRANAV_SENTENCES:
        # For manifold sentences, select a few representative concepts
        available_concepts = list(concept_prompts.keys())
        print(f"Available concepts in manifold dataset: {available_concepts}")
        
        preferred_concepts = ['animals', 'food', 'colors', 'clothing']
        selected_concepts = []
        for concept in preferred_concepts:
            if concept in available_concepts:
                selected_concepts.append(concept)
            if len(selected_concepts) >= 4:
                break
        
        # If we didn't find enough preferred concepts, fill with first available ones
        for concept in available_concepts:
            if concept not in selected_concepts:
                selected_concepts.append(concept)
            if len(selected_concepts) >= 4:
                break
        
        concept_prompt_pairs = []
        for concept in selected_concepts:
            concept_prompt_pairs.append({
                "concept": concept,
                "system_prompt": "You are a helpful assistant.",
                "user_prompt": f"Please write a sentence about {concept}."
            })
        
        print(f"Selected concepts for analysis: {[pair['concept'] for pair in concept_prompt_pairs]}")
    else:
        # Original prompts.json logic
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
        
        if concept not in concept_prompts:
            print(f"Concept '{concept}' not found in prompts.json. Available concepts: {list(concept_prompts.keys())}")
            print("Skipping this concept-prompt pair.\n")
            continue

        for target_layer in TARGET_LAYERS:
            print("\n" + "="*80)
            print(f"### ANALYZING LAYER {target_layer} for concept '{concept}' ###")
            print("="*80 + "\n")

            system_prompt_for_manifold = system_prompt if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
            
            concept_activations = get_final_token_activations(
                model, tokenizer, concept_prompts[concept], target_layer, 
                system_prompt=system_prompt_for_manifold
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_activations = {concept: concept_activations}
            analysis_results = analyse_manifolds(all_activations, local_centre=LOCAL_CENTRE)
            
            if concept not in analysis_results:
                print(f"Analysis for concept '{concept}' failed for layer {target_layer}. Skipping to next layer.")
                continue
            
            # Store analysis results for cross-concept perturbation
            if target_layer not in all_concept_analyses:
                all_concept_analyses[target_layer] = {}
            all_concept_analyses[target_layer][concept] = analysis_results[concept]

            if concept == "dog":
                dog_analysis = analysis_results[concept]
                dog_avg_eigenvalues[target_layer] = dog_analysis["eigenvalues"].mean().item()
                dog_top_eigenvectors[target_layer] = dog_analysis["eigenvectors"][0]

            # Global PC computation - COMMENTED OUT
            # if RUN_GLOBAL_PC_ANALYSIS and target_layer not in global_analysis_cache:
            #     print("\n" + "~"*80)
            #     print(f"### COMPUTING ENHANCED GLOBAL PCs FOR LAYER {target_layer} ###")
            #     print(f"### (prompts.json + STSB dataset for broader representation) ###")
            #     print("~"*80 + "\n")
            #     
            #     try:
            #         global_activations, prompt_to_source, all_prompts = get_enhanced_global_activations(
            #             model, tokenizer, concept_prompts, target_layer, 
            #             system_prompt=system_prompt_for_manifold
            #         )
            #         
            #         global_analysis = analyse_global_manifolds(global_activations)
            #         global_analysis_cache[target_layer] = {
            #             'analysis': global_analysis,
            #             'all_prompts': all_prompts,
            #             'prompt_to_source': prompt_to_source
            #         }
            #         
            #         gc.collect()
            #         if torch.cuda.is_available():
            #             torch.cuda.empty_cache()
            #         
            #         print(f"Global PC analysis cached for layer {target_layer}")
            #         
            #     except Exception as e:
            #         print(f"ERROR: Failed to compute global PC analysis for layer {target_layer}: {e}")
            #         print("Skipping global PC analysis for this layer.")
            #         import traceback
            #         traceback.print_exc()
            #         continue
            #     
            #     print("~"*80 + "\n")

            # Cosine similarity analysis - COMMENTED OUT
            # if RUN_GLOBAL_PC_ANALYSIS and target_layer in global_analysis_cache:
            #     global_data = global_analysis_cache[target_layer]
            #     global_analysis = global_data['analysis']
            #     
            #     try:
            #         cosine_sim_matrix = compute_pc_cosine_similarity(
            #             analysis_results[concept], 
            #             global_analysis, 
            #             concept, 
            #             target_layer, 
            #             top_k=5
            #         )
            #     except Exception as e:
            #         print(f"ERROR: Failed to compute cosine similarity for {concept} at layer {target_layer}: {e}")
            #         print("Skipping cosine similarity analysis for this concept-layer combination.")
            #         import traceback
            #         traceback.print_exc()

            print(f"\nRunning experiments for:")
            print(f"System prompt: '{system_prompt}'")
            print(f"User prompt: '{user_prompt}'")
            print(f"Concept: '{concept}'")
            print(f"Layer: {target_layer}")

            messages_to_test = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Skip standard experiments if CROSS_CONCEPT_ONLY is enabled
            if not CROSS_CONCEPT_ONLY:
                run_perturbation_experiment(
                    model, tokenizer, messages_to_test, target_layer, 
                    analysis_results[concept], concept, 
                    target_token_idx=None, perturb_once=PERTURB_ONCE, 
                    orthogonal_mode=False, use_largest_eigenvalue=True
                )
                
                # Top prompts analysis - only first PC (PC0)
                axis = 0
                if axis < len(analysis_results[concept]["eigenvectors"]):
                    pc_direction = analysis_results[concept]["eigenvectors"][axis]
                    
                    print("\n" + "="*80)
                    print(f"--- Analyzing original dataset prompts along PC{axis} '{concept}' direction (Layer {target_layer}) ---")
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
                
                # Orthogonal perturbation
                run_perturbation_experiment(
                    model, tokenizer, messages_to_test, target_layer,
                    analysis_results[concept], concept,
                    target_token_idx=None, perturb_once=PERTURB_ONCE,
                    orthogonal_mode=True, use_largest_eigenvalue=True
                )

                # Ablation experiment
                run_ablation_experiment(
                    model, tokenizer, messages_to_test, target_layer,
                    analysis_results[concept], concept,
                    target_token_idx=None, perturb_once=PERTURB_ONCE
                )
            else:
                print(f"\n CROSS_CONCEPT_ONLY mode enabled - skipping standard PC perturbations, top prompts analysis, orthogonal perturbations, and ablation experiments for '{concept}' at layer {target_layer}")

            # Global PC experiments - COMMENTED OUT
            # if RUN_GLOBAL_PC_ANALYSIS and target_layer in global_analysis_cache:
            #     global_data = global_analysis_cache[target_layer]
            #     global_analysis = global_data['analysis']
            #     all_prompts = global_data['all_prompts']
            #     
            #     global_analysis_for_helpers = {}
            #     for key, tensor in global_analysis.items():
            #         if torch.is_tensor(tensor):
            #             global_analysis_for_helpers[key] = tensor.cpu().to(dtype=torch.float16)
            #         else:
            #             global_analysis_for_helpers[key] = tensor
            #     
            #     print("\n" + "@"*80)
            #     print(f"### GLOBAL PC PERTURBATION EXPERIMENTS FOR LAYER {target_layer} ###")
            #     print(f"### Using Global PCs computed from {len(all_prompts)} sentences ###")
            #     print(f"### (prompts.json + STSB dataset for enhanced global representation) ###")
            #     print("@"*80 + "\n")
            #     
            #     print(f"Running GLOBAL PC perturbation experiments for concept '{concept}'...")
            #     try:
            #         run_perturbation_experiment(
            #             model, tokenizer, messages_to_test, target_layer, 
            #             global_analysis_for_helpers, f"GLOBAL-{concept}", 
            #             target_token_idx=None, perturb_once=PERTURB_ONCE, 
            #             orthogonal_mode=False, use_largest_eigenvalue=True
            #         )
            #     except Exception as e:
            #         print(f"ERROR in global PC perturbation experiment: {e}")
            #         print("Continuing with other analyses...")
            #     
            #     # Global PC prompt analysis - only first PC (PC0)
            #     axis = 0
            #     if axis < len(global_analysis["eigenvectors"]):
            #             
            #         global_pc_direction = global_analysis["eigenvectors"][axis]
            #         
            #         print("\n" + "="*80)
            #         print(f"--- Analyzing ALL dataset prompts along GLOBAL PC{axis} direction (Layer {target_layer}) ---")
            #         if USE_NORMALIZED_PROJECTION:
            #             print("Using normalized projections (projection magnitude / vector magnitude)")
            #         else:
            #             print("Using raw projections")
            #         
            #         prompt_sources = global_data.get('prompt_to_source', None)
            #         
            #         try:
            #             top_prompts_global = find_top_prompts_global(
            #                 all_prompts,
            #                 global_analysis["centered_acts"],
            #                 global_pc_direction,
            #                 n=10,
            #                 use_normalized_projection=USE_NORMALIZED_PROJECTION,
            #                 prompt_sources=prompt_sources
            #             )
            # 
            #             print(f"\nTop 10 sentences most aligned with POSITIVE GLOBAL PC{axis} direction:")
            #             for i, item in enumerate(top_prompts_global['positive'], 1):
            #                 if isinstance(item, tuple):
            #                     prompt, source = item
            #                     source_display = source.replace('concept_', '').replace('stsb_dataset', 'STSB')
            #                     print(f"{i:2d}. [{source_display}] '{prompt}'")
            #                 else:
            #                     print(f"{i:2d}. '{item}'")
            #                 
            #             print(f"\nTop 10 sentences most aligned with NEGATIVE GLOBAL PC{axis} direction:")
            #             for i, item in enumerate(top_prompts_global['negative'], 1):
            #                 if isinstance(item, tuple):
            #                     prompt, source = item
            #                     source_display = source.replace('concept_', '').replace('stsb_dataset', 'STSB')
            #                     print(f"{i:2d}. [{source_display}] '{prompt}'")
            #                 else:
            #                     print(f"{i:2d}. '{item}'")
            #                     
            #         except Exception as e:
            #             print(f"ERROR in global PC prompt analysis for axis {axis}: {e}")
            #             print("Continuing with next axis...")
            #             
            #         print("="*80)
            #     
            #     # Global PC orthogonal perturbation
            #     print(f"\nRunning GLOBAL PC orthogonal perturbation for concept '{concept}'...")
            #     try:
            #         run_perturbation_experiment(
            #             model, tokenizer, messages_to_test, target_layer,
            #             global_analysis_for_helpers, f"GLOBAL-{concept}",
            #             target_token_idx=None, perturb_once=PERTURB_ONCE,
            #             orthogonal_mode=True, use_largest_eigenvalue=True
            #         )
            #     except Exception as e:
            #         print(f"ERROR in global PC orthogonal perturbation: {e}")
            #         print("Continuing with other analyses...")
            # 
            #     # Global PC ablation
            #     print(f"\nRunning GLOBAL PC ablation experiment for concept '{concept}'...")
            #     try:
            #         run_ablation_experiment(
            #             model, tokenizer, messages_to_test, target_layer,
            #             global_analysis_for_helpers, f"GLOBAL-{concept}",
            #             target_token_idx=None, perturb_once=PERTURB_ONCE
            #         )
            #     except Exception as e:
            #         print(f"ERROR in global PC ablation experiment: {e}")
            #         print("Continuing with other analyses...")
            #     
            #     print("@"*80 + "\n")

    # Run cross-concept perturbation experiments
    print("\n" + "#"*80)
    print("### CROSS-CONCEPT PERTURBATION EXPERIMENTS ###")
    print("#"*80 + "\n")
    
    for target_layer in TARGET_LAYERS:
        if target_layer not in all_concept_analyses:
            continue
            
        layer_analyses = all_concept_analyses[target_layer]
        available_concepts = list(layer_analyses.keys())
        
        if len(available_concepts) < 2:
            print(f"Skipping cross-concept perturbation for layer {target_layer}: need at least 2 concepts, found {len(available_concepts)}")
            continue
            
        print(f"\n" + ">"*80)
        print(f"### CROSS-CONCEPT PERTURBATION FOR LAYER {target_layer} ###")
        print(f"### Available concepts: {available_concepts} ###")
        print(">"*80 + "\n")
        
        # For each concept, select up to 3 other concepts as targets
        for source_concept in available_concepts:
            # Get other concepts (excluding the source)
            other_concepts = [c for c in available_concepts if c != source_concept]
            
            # Select up to 3 target concepts
            target_concepts = other_concepts[:3]  # Take first 3 other concepts
            
            print(f"\n--- Cross-concept perturbations FROM '{source_concept.upper()}' ---")
            print(f"Target concepts: {target_concepts}")
            
            # Create messages for the source concept
            source_messages = []
            for pair in concept_prompt_pairs:
                if pair["concept"] == source_concept:
                    source_messages = [
                        {"role": "system", "content": pair["system_prompt"]},
                        {"role": "user", "content": pair["user_prompt"]}
                    ]
                    break
            
            if not source_messages:
                print(f"ERROR: Could not find messages for source concept '{source_concept}', skipping...")
                continue
            
            # Run perturbations to each target concept
            for target_concept in target_concepts:
                print(f"\nRunning cross-concept perturbation: {source_concept.upper()} → {target_concept.upper()}")
                try:
                    run_cross_concept_perturbation(
                        model, tokenizer, source_messages, target_layer,
                        layer_analyses[source_concept], layer_analyses[target_concept],
                        source_concept, target_concept, 
                        target_token_idx=None, perturb_once=PERTURB_ONCE
                    )
                except Exception as e:
                    print(f"ERROR in {source_concept} → {target_concept} cross-concept perturbation: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"--- End of '{source_concept.upper()}' perturbations ---")
        
        print(">"*80 + "\n")

    # Plotting removed per user request
    # print("\n" + "#"*80)
    # print("### PLOTTING OVERALL RESULTS ###")
    # print("#"*80 + "\n")
    # plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "lastToken")
    # plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "lastToken")

if __name__ == "__main__":
    main()