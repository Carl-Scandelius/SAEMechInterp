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
    MODEL_NAME,
    DEVICE,
)
from transformers import logging
logging.set_verbosity(40)

USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = False

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

def main():
    print(f"\nConfiguration: PERTURB_ONCE is set to {PERTURB_ONCE}\n")
    print(f"Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD is set to {USE_SYSTEM_PROMPT_FOR_MANIFOLD}\n")

    dog_avg_eigenvalues = {}
    dog_top_eigenvectors = {}
    model_name_str = MODEL_NAME.split('/')[-1]

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    with open('prompts.json', 'r', encoding='utf-8') as f:
        concept_prompts = json.load(f)
    
    TARGET_LAYERS = [0, 15, 31] # Llama-3.1-8B has 32 layers (0-31)
    AXES_TO_ANALYZE = range(5)

    for target_layer in TARGET_LAYERS:
        print("\n" + "#"*80)
        print(f"### STARTING ANALYSIS FOR LAYER {target_layer} ###")
        print("#"*80 + "\n")

        all_activations = {}
        system_prompt = "You are a language model assistant. Please translate the following text accurately from English into German:"
        system_prompt_for_manifold = system_prompt if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""

        for concept, prompts in concept_prompts.items():
            all_activations[concept] = get_final_token_activations(model, tokenizer, prompts, target_layer, system_prompt=system_prompt_for_manifold)
            gc.collect()
            torch.cuda.empty_cache()

        analysis_results = analyse_manifolds(all_activations)

        if "dog" in analysis_results:
            dog_analysis = analysis_results["dog"]
            dog_avg_eigenvalues[target_layer] = dog_analysis["eigenvalues"].mean().item()
            dog_top_eigenvectors[target_layer] = dog_analysis["eigenvectors"][0]
        
        test_concept = "dog"
        user_prompt = "The dog was running around the park. It was a labrador."
        system_prompt = "You are a language model assistant. Please translate the following text accurately from English into German:"

        if test_concept not in analysis_results:
            print(f"Analysis for concept '{test_concept}' not available for layer {target_layer}. Skipping to next layer.")
            continue

        messages_to_test = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        run_perturbation_experiment(
            model, tokenizer, messages_to_test, target_layer, 
            analysis_results[test_concept], test_concept, AXES_TO_ANALYZE, 
            target_token_idx=None, perturb_once=PERTURB_ONCE, 
            orthogonal_mode=False, use_largest_eigenvalue=True  # actual orthogonal eigenval too small
        )
        
        # Display top prompts aligned with the PC direction
        for axis in AXES_TO_ANALYZE:
            if axis >= len(analysis_results[test_concept]["eigenvectors"]):
                break
                
            pc_direction = analysis_results[test_concept]["eigenvectors"][axis]
            
            print("\n" + "="*80)
            print(f"--- Analyzing original dataset prompts along the PC{axis} '{test_concept}' direction (Layer {target_layer}) ---")
            top_prompts = find_top_prompts(
                concept_prompts[test_concept],
                analysis_results[test_concept]["centered_acts"],
                pc_direction,
                n=10
            )

            print(f"\nTop 10 prompts most aligned with POSITIVE PC{axis} direction:")
            for i, prompt in enumerate(top_prompts['positive'], 1):
                print(f"{i:2d}. '{prompt}'")
                
            print(f"\nTop 10 prompts most aligned with NEGATIVE PC{axis} direction:")
            for i, prompt in enumerate(top_prompts['negative'], 1):
                print(f"{i:2d}. '{prompt}'")
            print("="*80)
        
        # --- PROJECTION-BASED PERTURBATION ---
        # For each PC, run a projection-based perturbation that zeroes out all other PCs
        from helpers import run_projection_based_perturbation
        for axis in AXES_TO_ANALYZE:
            if axis >= len(analysis_results[test_concept]["eigenvectors"]):
                break
                
            # Run projection-based perturbation for this PC
            run_projection_based_perturbation(
                model, tokenizer, messages_to_test, target_layer,
                analysis_results[test_concept], test_concept, axis,
                target_token_idx=None, perturb_once=PERTURB_ONCE
            )

        # --- ORTHOGONAL PERTURBATION ---
        run_perturbation_experiment(
            model, tokenizer, messages_to_test, target_layer,
            analysis_results[test_concept], test_concept,
            target_token_idx=None, perturb_once=PERTURB_ONCE,
            orthogonal_mode=True, use_largest_eigenvalue=True
        )

        # --- ABLATION EXPERIMENT ---
        from helpers import run_ablation_experiment
        run_ablation_experiment(
            model, tokenizer, messages_to_test, target_layer,
            analysis_results[test_concept], test_concept,
            target_token_idx=None, perturb_once=PERTURB_ONCE
        )

    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "lastToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "lastToken")

if __name__ == "__main__":
    main()