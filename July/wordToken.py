"""
Word manifold exploration based on specific concept words.

This script adapts the approach from lastToken.py. Instead of using the final
token's embedding, it identifies the token/s corresponding to a specific concept
word within each prompt. It then uses the embedding of the last
token of that word (cf. causal attention) for manifold analysis.
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
logging.set_verbosity(logging.ERROR)

USE_SYSTEM_PROMPT_FOR_MANIFOLD = False 
concept_keywords = {
    "dog": ["dog", "dogs", "dog's", "puppy", "puppies"],
    "lion": ["lion", "lions", "lion's"],
    "human": ["human", "humans", "human's", "man", "woman", "person", "people"],
    "house": ["house", "houses", "house's", "home"]
}

def find_word_token_index(prompt, concept, tokenizer, add_generation_prompt):
    """Finds the index of the last token of the last occurrence of a concept word in a prompt."""
    variations = concept_keywords.get(concept, [concept])

    lower_prompt = prompt.lower()
    last_occurrence_pos = -1
    word_found_len = -1
    for var in variations:
        pos = lower_prompt.rfind(var)
        if pos > last_occurrence_pos:
            last_occurrence_pos = pos
            word_found_len = len(var)

    if last_occurrence_pos == -1:
        return -1, None

    char_start = last_occurrence_pos
    char_end = char_start + word_found_len

    messages = [{"role": "user", "content": prompt}]
    text_for_model = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    prompt_start_in_text = text_for_model.lower().find(lower_prompt)
    if prompt_start_in_text == -1:
        return -1, None

    final_char_start = prompt_start_in_text + char_start
    final_char_end = prompt_start_in_text + char_end

    inputs = tokenizer(text_for_model, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop('offset_mapping').squeeze(0)

    token_indices_for_word = []
    for i, (start, end) in enumerate(offset_mapping):
        if max(start, final_char_start) < min(end, final_char_end):
            token_indices_for_word.append(i)

    if not token_indices_for_word:
        return -1, None

    target_token_idx = max(token_indices_for_word)
    inputs.to(DEVICE)
    return target_token_idx, inputs

def get_word_token_activations(model, tokenizer, prompt_keyword_pairs, layer_idx, concept, system_prompt=""):
    activations = []
    print(f"Extracting activations from layer {layer_idx} for concept '{concept}'...")

    activation_storage = []
    def hook_fn(module, input, output, target_token_idx):
        token_activation = output[0][:, target_token_idx, :].detach().cpu()
        activation_storage.append(token_activation)

    for prompt, keyword in tqdm(prompt_keyword_pairs, desc=f"Extracting '{concept}' activations"):
        activation_storage.clear()

        messages = [{'role': 'user', 'content': prompt}]
        if USE_SYSTEM_PROMPT_FOR_MANIFOLD and system_prompt:
            messages.insert(0, {'role': 'system', 'content': system_prompt})

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(DEVICE)
        
        input_ids_list = inputs[0].tolist()
        keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
        
        target_token_idx = -1
        for i in range(len(input_ids_list) - len(keyword_ids) + 1):
            if input_ids_list[i:i+len(keyword_ids)] == keyword_ids:
                target_token_idx = i + len(keyword_ids) - 1
        
        if target_token_idx == -1:
            continue

        hook_handle = model.model.layers[layer_idx].register_forward_hook(
            lambda module, input, output: hook_fn(module, input, output, target_token_idx)
        )
        
        with torch.no_grad():
            model(input_ids=inputs)
        
        hook_handle.remove()

        if activation_storage:
            activations.append(activation_storage[0])

    if not activations:
        print(f"Warning: Could not extract any activations for concept '{concept}' in layer {layer_idx}.")
        return torch.empty(0, model.config.hidden_size, device=DEVICE)

    return torch.cat(activations, dim=0)

def main():
    dog_avg_eigenvalues = {}
    dog_top_eigenvectors = {}
    model_name_str = MODEL_NAME.split('/')[-1]

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    with open('prompts.json', 'r', encoding='utf-8') as f:
        concept_prompts = json.load(f)

    filtered_prompts = {}
    print("Filtering prompts based on keywords...")
    for concept, prompts in concept_prompts.items():
        if concept in concept_keywords:
            keywords = concept_keywords[concept]
            pairs = []
            for p in prompts:
                for keyword in keywords:
                    if keyword in p.lower():
                        pairs.append((p, keyword))
                        break
            filtered_prompts[concept] = pairs
        else:
            print(f"Warning: No keywords defined for concept '{concept}'. Falling back to simple string match.")
            filtered_prompts[concept] = [(p, concept) for p in prompts if concept in p.lower()]
        
        print(f"  - Concept '{concept}': Found {len(filtered_prompts[concept])} matching prompts out of {len(prompts)}.")

    TARGET_LAYERS = [0, 15, 31] # Llama-3.1-8B has 32 layers (0-31)
    AXES_TO_ANALYZE = range(5)

    for target_layer in TARGET_LAYERS:
        print("\n" + "#"*80)
        print(f"### STARTING ANALYSIS FOR LAYER {target_layer} ###")
        print("#"*80 + "\n")

        all_activations = {}
        system_prompt_for_manifold = "You are a language model assistant. Please translate the following text accurately from English into German:" if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
        for concept, prompts in filtered_prompts.items():
            if not prompts:
                print(f"No prompts for concept '{concept}' after filtering. Skipping.")
                continue
        for concept, prompt_keyword_pairs in filtered_prompts.items():
            all_activations[concept] = get_word_token_activations(
                model, tokenizer, prompt_keyword_pairs, target_layer, concept, system_prompt=system_prompt_for_manifold
            )
            gc.collect()
            torch.cuda.empty_cache()

        analysis_results = analyse_manifolds(all_activations)

        if "dog" in analysis_results:
            dog_analysis = analysis_results["dog"]
            # Store data for plotting
            dog_avg_eigenvalues[target_layer] = dog_analysis["eigenvalues"].mean().item()
            dog_top_eigenvectors[target_layer] = dog_analysis["eigenvectors"][0]
        
        test_concept = "dog"
        user_prompt = "The dog was running around the park. It was a labrador."
        system_prompt = "You are a language model assistant. Please translate the following text accurately from English into German:"

        messages_to_test = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs_for_gen = tokenizer.apply_chat_template(
            messages_to_test,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        decoded_input = tokenizer.decode(inputs_for_gen[0], skip_special_tokens=False)
        decoded_input_lower = decoded_input.lower()
        variations = concept_keywords.get(test_concept, [test_concept])
        
        last_pos = -1
        matched_variation = None
        for var in variations:
            pos = decoded_input_lower.rfind(var.lower())
            if pos > last_pos:
                last_pos = pos
                matched_variation = var
        
        if last_pos == -1:
            print(f"Debug: Concept '{test_concept}' variations {variations} not found in decoded text: {decoded_input[:100]}...")
            target_token_idx = -1
        else:
            input_ids = inputs_for_gen[0].tolist()
            keyword_end_pos = last_pos + len(matched_variation)
            token_indices = []
            
            for i, token_id in enumerate(input_ids):
                # Get the text for this token
                for var in variations:
                    if var.lower() in token_text:
                        token_indices.append(i)
            
            if token_indices:
                target_token_idx = max(token_indices)
            else:
                # Fallback: try the original exact token matching approach
                concept_ids = tokenizer.encode(test_concept, add_special_tokens=False)
                for i in range(len(input_ids) - len(concept_ids) + 1):
                    if input_ids[i:i+len(concept_ids)] == concept_ids:
                        target_token_idx = i + len(concept_ids) - 1
        
        if target_token_idx == -1:
            print(f"Warning: Concept '{test_concept}' not found in the tokenized test prompt. Skipping perturbation tests for layer {target_layer}.")
            continue

        if test_concept not in analysis_results:
            print(f"No analysis results for concept '{test_concept}', skipping layer {target_layer}.")
            continue
        
        run_perturbation_experiment(
            model, tokenizer, messages_to_test, target_layer,
            analysis_results[test_concept], test_concept, AXES_TO_ANALYZE,
            target_token_idx=target_token_idx, perturb_once=True  # Always perturb once for word tokens
        )
        
        for axis in AXES_TO_ANALYZE:
            if axis >= len(analysis_results[test_concept]["eigenvectors"]):
                break
                
            pc_direction = analysis_results[test_concept]["eigenvectors"][axis]
            
            print("\n" + "="*80)
            print(f"--- Analyzing original dataset prompts along the PC{axis} '{test_concept}' direction (Layer {target_layer}) ---")
            prompts_for_concept = [p for p, k in filtered_prompts[test_concept]]
            top_prompts_dict = find_top_prompts(
                prompts_for_concept,
                analysis_results[test_concept]["centered_acts"],
                pc_direction,
                n=10
            )

            print(f"\nTop 10 prompts most aligned with POSITIVE PC{axis} direction:")
            for i, prompt in enumerate(top_prompts_dict['positive'], 1):
                print(f"{i:2d}. '{prompt}'")
            
            print(f"\nTop 10 prompts most aligned with NEGATIVE PC{axis} direction:")
            for i, prompt in enumerate(top_prompts_dict['negative'], 1):
                print(f"{i:2d}. '{prompt}'")
            print("="*80)

        # --- ORTHOGONAL PERTURBATION ---
        # Run orthogonal perturbation experiment if we have results for this concept
        if test_concept in analysis_results:
            run_perturbation_experiment(
                model, tokenizer, inputs_for_gen, target_layer,
                analysis_results[test_concept], test_concept,
                target_token_idx=target_token_idx, perturb_once=True,  # Always perturb once for word tokens
                orthogonal_mode=True, use_largest_eigenvalue=False  # Use actual orthogonal eigenvalue as in original
            )
        else:
            print("\n" + "="*80)
            print(f"--- ORTHOGONAL PERTURBATION ON CONCEPT: '{test_concept}' ---")
            print(f"--- LAYER: {target_layer} ---")
            print("="*80)
            print(f"No analysis results for concept '{test_concept}', skipping orthogonal perturbation.")

    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "wordToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "wordToken")

if __name__ == "__main__":
    main()
