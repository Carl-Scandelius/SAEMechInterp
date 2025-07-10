"""Transformer concept manifold exploration and perturbation analysis."""

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
    run_projection_based_perturbation,
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

def get_final_token_activations(
    model, tokenizer, prompts: List[str], layer_idx: int, system_prompt: str = ""
) -> torch.Tensor:
    """Extract final token activations from specified layer."""
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

def get_global_activations(
    model, tokenizer, concept_prompts: Dict[str, List[str]], layer_idx: int, system_prompt: str = ""
) -> Tuple[torch.Tensor, List[str], List[str]]:
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
    model, tokenizer, concept_prompts: Dict[str, List[str]], layer_idx: int, system_prompt: str = ""
) -> Tuple[torch.Tensor, List[str], List[str]]:
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

def analyse_global_manifolds(global_activations: torch.Tensor) -> Dict[str, Any]:
    """Compute global PCs via SVD from combined activations."""
    print(f"Computing global PCs from {global_activations.shape[0]} activations...")
    print(f"Tensor device: {global_activations.device}, dtype: {global_activations.dtype}")
    
    original_dtype = global_activations.dtype
    original_device = global_activations.device
    
    if global_activations.dtype == torch.float16:
        print("Converting from float16 to float32 for SVD computation...")
        global_activations = global_activations.float()
    
    device = global_activations.device
    
    mean_activation = global_activations.mean(dim=0)
    centered_activations = global_activations - mean_activation
    
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
                U, S, Vt = U.to(device), S.to(device), Vt.to(device)
                centered_activations = centered_activations_cpu.to(device)
                print("SVD completed on CPU, results moved back to GPU")
            except RuntimeError as e2:
                print(f"SVD also failed on CPU: {e2}")
                print("Trying with double precision...")
                centered_activations_double = centered_activations_cpu.double()
                U, S, Vt = torch.svd(centered_activations_double)
                U, S, Vt = U.float().to(device), S.float().to(device), Vt.float().to(device)
                centered_activations = centered_activations_double.float().to(device)
                print("SVD completed with double precision")
        else:
            raise e
    
    eigenvalues = S ** 2 / (global_activations.shape[0] - 1)
    eigenvectors = Vt.T
    
    global_analysis = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors.T,  # Each row is an eigenvector
        "centered_acts": centered_activations,
        "mean": mean_activation,
        "_original_dtype": original_dtype,
        "_original_device": original_device
    }
    
    print(f"Global PCA completed on {device}. Top 5 eigenvalues: {eigenvalues[:5].tolist()}")
    
    return global_analysis

def find_top_prompts_global(
    all_prompts: List[str], 
    global_centered_acts: torch.Tensor, 
    pc_direction: torch.Tensor, 
    n: int = 10, 
    use_normalized_projection: bool = True, 
    prompt_sources: Optional[List[str]] = None
) -> Dict[str, List]:
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
    concept_analysis: Dict[str, Any], 
    global_analysis: Dict[str, Any], 
    concept_name: str, 
    layer_idx: int, 
    top_k: int = 5
) -> torch.Tensor:
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
    model, tokenizer, messages: List[Dict[str, str]], layer_idx: int,
    source_concept_analysis: Dict[str, Any], target_concept_analysis: Dict[str, Any],
    source_concept_name: str, target_concept_name: str,
    target_token_idx: Optional[int] = None, perturb_once: bool = False
) -> None:
    """
    Perturb along the direction between two concept manifolds.
    """
    # Get centroids (mean activations) for both concepts
    source_centroid = source_concept_analysis["mean"].to(DEVICE)
    target_centroid = target_concept_analysis["mean"].to(DEVICE)
    
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
    
    print("\n" + "="*80)
    print(f"--- CROSS-CONCEPT PERTURBATION: '{source_concept_name}' → '{target_concept_name}' ---")
    print(f"--- LAYER: {layer_idx} ---")
    print(f"--- Cross-concept distance: {cross_concept_distance:.4f} ---")
    print("="*80)
    
    print(f"\nSystem: '{system_prompt}'")
    print(f"User: '{user_prompt}'")
    
    # Generate original output (0% perturbation)
    original_output = generate_with_perturbation(
        model, tokenizer, inputs, layer_idx, direction_vector, 0.0, 
        torch.tensor(1.0, device=model.device), target_token_idx, perturb_once
    )
    print(f"Original (0%): {original_output}")
    
    # Perturb in steps of 0.2 from 0.2 to 1.0
    print(f"\n--- Perturbing along {source_concept_name} → {target_concept_name} direction ---")
    for step in range(1, 6):  # 1, 2, 3, 4, 5 corresponding to 0.2, 0.4, 0.6, 0.8, 1.0
        percentage = step * 0.2
        
        # Perturbation magnitude: step * 0.2 * cross_concept_distance
        perturbation_magnitude = percentage * cross_concept_distance
        
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, layer_idx, direction_vector, 
            perturbation_magnitude, torch.tensor(1.0, device=model.device),
            target_token_idx, perturb_once
        )
        
        print(f"Step {step} ({percentage*100:.0f}%): {perturbed_output}")
    
    print("="*80)

def main():
    print(f"\nConfiguration: PERTURB_ONCE={PERTURB_ONCE}")
    print(f"Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD={USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
    print(f"Configuration: USE_NORMALIZED_PROJECTION={USE_NORMALIZED_PROJECTION}")
    print(f"Configuration: RUN_GLOBAL_PC_ANALYSIS={RUN_GLOBAL_PC_ANALYSIS}\n")

    dog_avg_eigenvalues = {}
    dog_top_eigenvectors = {}
    global_analysis_cache = {}
    # Store analysis results for cross-concept perturbation
    all_concept_analyses = {}
    model_name_str = MODEL_NAME.split('/')[-1]

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    with open('prompts.json', 'r', encoding='utf-8') as f:
        concept_prompts = json.load(f)
    
    TARGET_LAYERS = [0, 15, 31]
    AXES_TO_ANALYZE = range(5)

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
            analysis_results = analyse_manifolds(all_activations)
            
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

            run_perturbation_experiment(
                model, tokenizer, messages_to_test, target_layer, 
                analysis_results[concept], concept, AXES_TO_ANALYZE, 
                target_token_idx=None, perturb_once=PERTURB_ONCE, 
                orthogonal_mode=False, use_largest_eigenvalue=True
            )
            
            # Top prompts analysis
            for axis in AXES_TO_ANALYZE:
                if axis >= len(analysis_results[concept]["eigenvectors"]):
                    break
                    
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
            #             global_analysis_for_helpers, f"GLOBAL-{concept}", AXES_TO_ANALYZE, 
            #             target_token_idx=None, perturb_once=PERTURB_ONCE, 
            #             orthogonal_mode=False, use_largest_eigenvalue=True
            #         )
            #     except Exception as e:
            #         print(f"ERROR in global PC perturbation experiment: {e}")
            #         print("Continuing with other analyses...")
            #     
            #     # Global PC prompt analysis
            #     for axis in AXES_TO_ANALYZE:
            #         if axis >= len(global_analysis["eigenvectors"]):
            #             break
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
        
        # Check if we have both dog and lion analyses for this layer
        if "dog" in layer_analyses and "lion" in layer_analyses:
            print(f"\n" + ">"*80)
            print(f"### CROSS-CONCEPT PERTURBATION FOR LAYER {target_layer} ###")
            print(">"*80 + "\n")
            
            # Test messages for cross-concept perturbation
            dog_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please write a sentence about dogs."}
            ]
            lion_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please write a sentence about lions."}
            ]
            
            # Dog → Lion perturbation
            print(f"Running cross-concept perturbation: DOG → LION")
            try:
                run_cross_concept_perturbation(
                    model, tokenizer, dog_messages, target_layer,
                    layer_analyses["dog"], layer_analyses["lion"],
                    "dog", "lion", target_token_idx=None, perturb_once=PERTURB_ONCE
                )
            except Exception as e:
                print(f"ERROR in dog → lion cross-concept perturbation: {e}")
                import traceback
                traceback.print_exc()
            
            # Lion → Dog perturbation
            print(f"\nRunning cross-concept perturbation: LION → DOG")
            try:
                run_cross_concept_perturbation(
                    model, tokenizer, lion_messages, target_layer,
                    layer_analyses["lion"], layer_analyses["dog"],
                    "lion", "dog", target_token_idx=None, perturb_once=PERTURB_ONCE
                )
            except Exception as e:
                print(f"ERROR in lion → dog cross-concept perturbation: {e}")
                import traceback
                traceback.print_exc()
            
            print(">"*80 + "\n")
        else:
            missing_concepts = []
            if "dog" not in layer_analyses:
                missing_concepts.append("dog")
            if "lion" not in layer_analyses:
                missing_concepts.append("lion")
            print(f"Skipping cross-concept perturbation for layer {target_layer}: missing analyses for {missing_concepts}")

    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "lastToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "lastToken")

if __name__ == "__main__":
    main()