"""
Word manifold exploration.

Pass concept-labelled prompts (in QA format) into decoder-only model (Llama 3.3, GPT-2, or something more modern). Extract corresponding token representation in residual stream of layer l.

We now have some sample of the representation space. Ideally this is in the form of concept manifolds. Centre this representation to remove correlation between centroids

Find 'effective' eigenvectors of the manifolds. Project work token (from new labelled prompt) representation onto respective 'effective' concept manifold.

Check decoded sentence for:
1) original prompt
2) original with final token embedding perturbed in PC1 by pm 1.5x eigenvalue, pm 2x eigenvalue, pm 5x eigenvalue

Find prompts in initial dataset that are furthest in the direction of PC1: how are they correlated? Is this a global feature or just for that manifold's PC1?

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
)

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = False  # Can be configured by runner script

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

def generate_with_perturbation(model, tokenizer, messages, layer_idx, direction, magnitude, eigenvalue, perturb_once=False):
    perturbation = direction * magnitude * torch.sqrt(eigenvalue)
    
    def hook_fn_modify(module, input, output):
        hidden_states = output[0]
        if perturb_once:
            # Only apply perturbation during the initial prompt processing pass
            if hidden_states.shape[1] > 1:
                hidden_states[:, -1, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        else:
            # Original behavior: apply perturbation on every forward pass
            hidden_states[:, -1, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        return (hidden_states,) + output[1:]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(DEVICE)

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_modify)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=70,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    hook_handle.remove()

    prompt_length = inputs.shape[1]
    decoded_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    
    return decoded_text

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

        for axis in AXES_TO_ANALYZE:
            print("\n" + "="*80)
            print(f"--- INTERVENTION EXPERIMENT ON CONCEPT: '{test_concept}' ---")
            print(f"--- LAYER: {target_layer}, AXIS: {axis} ---")
            print("="*80)

            concept_analysis = analysis_results[test_concept]

            if axis >= len(concept_analysis["eigenvectors"]):
                print(f"Axis {axis} is out of bounds for the number of principal components found ({len(concept_analysis['eigenvectors'])}).")
                print("Skipping remaining axes for this layer.")
                break

            pc_direction = concept_analysis["eigenvectors"][axis]
            pc_eigenvalue = concept_analysis["eigenvalues"][axis]
            
            print(f"\nSystem Prompt: '{system_prompt}'")
            print(f"User Prompt:   '{user_prompt}'")

            original_output = generate_with_perturbation(model, tokenizer, messages_to_test, target_layer, pc_direction, 0, pc_eigenvalue, perturb_once=PERTURB_ONCE)
            print(f"Original model completion: {original_output}")
            
            perturbation_scales = [-20.0, -10.0, -5.0, -2.5, -1.5, 0.0, 1.5, 2.5, 5.0, 10.0, 20.0]
            
            print(f"\n--- Perturbing final token activation along PC{axis} ---")
            for scale in perturbation_scales:
                perturbed_output = generate_with_perturbation(
                    model, tokenizer, messages_to_test, target_layer, pc_direction, scale, pc_eigenvalue,
                    perturb_once=PERTURB_ONCE
                )
                print(f"Perturbation scale {scale:+.1f}x: {perturbed_output}")

            print("\n" + "="*80)
            print(f"--- Analyzing original dataset prompts along the PC{axis} '{test_concept}' direction (Layer {target_layer}) ---")
            top_prompts = find_top_prompts(
                concept_prompts[test_concept],
                concept_analysis["centered_acts"],
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

        # --- ORTHOGONAL PERTURBATION ---
        print("\n" + "="*80)
        print(f"--- ORTHOGONAL PERTURBATION ON CONCEPT: '{test_concept}' ---")
        print(f"--- LAYER: {target_layer} ---")
        print("="*80)

        concept_analysis = analysis_results[test_concept]
        effective_mask = concept_analysis["effective_mask"]
        
        orthogonal_pc_index = -1
        for i, is_effective in enumerate(effective_mask):
            if not is_effective:
                orthogonal_pc_index = i
                break

        if orthogonal_pc_index != -1:
            ortho_direction = concept_analysis["eigenvectors"][orthogonal_pc_index]
            ortho_eigenvalue = concept_analysis["eigenvalues"][orthogonal_pc_index]

            print(f"Perturbing along first orthogonal direction (PC{orthogonal_pc_index})...")
            print(f"\nSystem Prompt: '{system_prompt}'")
            print(f"User Prompt:   '{user_prompt}'")

            original_output = generate_with_perturbation(model, tokenizer, messages_to_test, target_layer, ortho_direction, 0, concept_analysis["eigenvalues"][0], perturb_once=PERTURB_ONCE)
            print(f"Original model completion: {original_output}")

            for scale in perturbation_scales:
                perturbed_output = generate_with_perturbation(
                    model, tokenizer, messages_to_test, target_layer, ortho_direction, scale, concept_analysis["eigenvalues"][0],   #ortho_eigenvalue, (ortho eigenvalue is too small; I use largest now)
                    perturb_once=PERTURB_ONCE
                )
                print(f"Perturbation scale {scale:+.1f}x: {perturbed_output}")
        else:
            print("Could not find an orthogonal (ineffective) direction to perturb.")

    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "lastToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "lastToken")

if __name__ == "__main__":
    main()