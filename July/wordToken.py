"""
Word manifold exploration based on specific concept words.

This script adapts the approach from lastToken.py. Instead of using the final
token's embedding, it identifies the token/s corresponding to a specific concept
word within each prompt. It then uses the embedding of the last
token of that word (cf. causal attention) for manifold analysis.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def get_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def find_word_token_index(prompt, concept, tokenizer, add_generation_prompt):
    """Finds the index of the last token of the last occurrence of a concept word in a prompt."""
    variations = {
        "dog": ["dog", "dogs", "dog's"],
        "lion": ["lion", "lions", "lion's"],
        "human": ["human", "humans", "human's", "man", "woman", "person"],
        "house": ["house", "houses", "house's"]
    }.get(concept, [concept])

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

def get_word_token_activations(model, tokenizer, prompts, layer_idx, concept):
    activations = []
    print(f"Extracting activations from layer {layer_idx} for concept '{concept}'...")

    for prompt in tqdm(prompts, desc=f"Processing prompts for '{concept}'"):
        target_token_idx, inputs = find_word_token_index(prompt, concept, tokenizer, add_generation_prompt=False)

        if target_token_idx == -1:
            print(f"Warning: Could not find token for concept '{concept}' in prompt: '{prompt}'. Skipping.")
            continue

        activation_storage = []
        def hook_fn(module, input, output):
            token_activation = output[0][:, target_token_idx, :].detach().cpu()
            activation_storage.append(token_activation)

        hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        hook_handle.remove()

        if activation_storage:
            activations.append(activation_storage[0])

    if not activations:
        return torch.empty(0, model.config.hidden_size)

    return torch.cat(activations, dim=0)

def analyze_manifolds(all_activations_by_concept):
    """
    Centres the concept manifolds and finds their 'effective' eigenvectors using PCA.
    """
    concept_analysis = {}
    
    centroids = {
        concept: acts.mean(dim=0)
        for concept, acts in all_activations_by_concept.items()
    }
    global_centroid = torch.stack(list(centroids.values())).mean(dim=0)
    centered_activations = {
        concept: acts - global_centroid
        for concept, acts in all_activations_by_concept.items()
    }
    
    print("Finding effective eigenvectors via PCA...")
    for concept, acts in centered_activations.items():
        if acts.shape[0] < 2: # PCA needs at least 2 samples
            print(f"Concept '{concept}': Not enough samples ({acts.shape[0]}) for PCA. Skipping.")
            continue
        pca = PCA()
        pca.fit(acts.numpy())
        
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        mean_eigval = eigenvalues.mean()
        threshold = mean_eigval
        
        effective_mask = eigenvalues > threshold
        
        print(f"Concept '{concept}': Found {np.sum(effective_mask)} effective eigenvectors out of {len(eigenvectors)} (threshold: {threshold:.4f})")
        
        concept_analysis[concept] = {
            "pca": pca,
            "eigenvectors": torch.tensor(eigenvectors, dtype=acts.dtype),
            "eigenvalues": torch.tensor(eigenvalues, dtype=acts.dtype),
            "effective_mask": torch.tensor(effective_mask, dtype=torch.bool),
            "centered_acts": acts
        }
    
    return concept_analysis

def generate_with_perturbation(model, tokenizer, layer_idx, direction, magnitude, eigenvalue, target_token_idx, inputs):
    perturbation = direction * magnitude * torch.sqrt(eigenvalue)
    
    def hook_fn_modify(module, input, output):
        hidden_states = output[0]
        # Only apply perturbation during the initial prompt processing pass,
        # not during the single-token generation steps.
        # We can detect this by checking the sequence length.
        if hidden_states.shape[1] > 1:
            hidden_states[:, target_token_idx, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
        return (hidden_states,) + output[1:]

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_modify)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=70,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    hook_handle.remove()

    prompt_length = inputs['input_ids'].shape[1]
    decoded_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    
    return decoded_text

def find_top_prompts(prompts, centered_acts, direction, n=10):
    projections = centered_acts @ direction
    sorted_values, sorted_indices = torch.sort(projections, descending=False)
    neg_indices = sorted_indices[:n]
    pos_indices = sorted_indices[-n:].flip(dims=[0])
    top_positive_prompts = [prompts[i] for i in pos_indices]
    top_negative_prompts = [prompts[i] for i in neg_indices]

    return {
        "positive": top_positive_prompts,
        "negative": top_negative_prompts
    }

def plot_avg_eigenvalues(eigenvalue_data, model_name_str, filename_prefix):
    """Plots the average eigenvalue for a concept across layers.""" 
    if not eigenvalue_data:
        print("No eigenvalue data to plot.")
        return
    
    layers = sorted(eigenvalue_data.keys())
    avg_eigenvalues = [eigenvalue_data[l] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_eigenvalues, marker='o')
    plt.title(f'Average "Dog" Manifold Eigenvalue vs. Layer\nModel: {model_name_str}')
    plt.xlabel("Model Layer")
    plt.ylabel("Average Eigenvalue")
    plt.xticks(layers)
    plt.grid(True, linestyle='--', alpha=0.6)
    filename = f"{filename_prefix}_dog_avg_eigenvalue.png"
    plt.savefig(filename)
    print(f"Saved average eigenvalue plot to {filename}")
    plt.close()

def plot_similarity_matrix(eigenvector_data, model_name_str, filename_prefix):
    """Plots the cosine similarity matrix of top eigenvectors across layers.""" 
    if len(eigenvector_data) < 2:
        print("Not enough eigenvector data to create a similarity matrix.")
        return

    layers = sorted(eigenvector_data.keys())
    num_layers = len(layers)
    similarity_matrix = torch.zeros((num_layers, num_layers))

    eigenvectors = [eigenvector_data[l] for l in layers]

    for i in range(num_layers):
        for j in range(num_layers):
            sim = F.cosine_similarity(eigenvectors[i].unsqueeze(0), eigenvectors[j].unsqueeze(0))
            similarity_matrix[i, j] = sim.item()

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=layers, yticklabels=layers)
    plt.title(f'Cosine Similarity of "Dog" Manifold PC0 Across Layers\nModel: {model_name_str}')
    plt.xlabel("Model Layer")
    plt.ylabel("Model Layer")
    filename = f"{filename_prefix}_dog_pc0_similarity.png"
    plt.savefig(filename)
    print(f"Saved eigenvector similarity matrix to {filename}")
    plt.close()

def main():
    # Data for plotting across layers
    dog_avg_eigenvalues = {}
    dog_top_eigenvectors = {}
    model_name_str = MODEL_NAME.split('/')[-1]

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    with open('prompts.json', 'r', encoding='utf-8') as f:
        concept_prompts = json.load(f)

    # Filter prompts to only include those with the concept word
    filtered_prompts = {}
    variations = {
        "dog": ["dog", "dogs", "dog's"],
        "lion": ["lion", "lions", "lion's"],
        "human": ["human", "humans", "human's", "man", "woman", "person"],
        "house": ["house", "houses", "house's"]
    }
    for concept, prompts in concept_prompts.items():
        concept_vars = variations.get(concept, [concept])
        filtered_prompts[concept] = [
            p for p in prompts 
            if any(var in p.lower() for var in concept_vars)
        ]
        print(f"Concept '{concept}': Filtered from {len(prompts)} to {len(filtered_prompts[concept])} prompts.")

    TARGET_LAYERS = [0, 15, 31] # Llama-3.1-8B has 32 layers (0-31)
    AXES_TO_ANALYZE = range(5)

    for target_layer in TARGET_LAYERS:
        print("\n" + "#"*80)
        print(f"### STARTING ANALYSIS FOR LAYER {target_layer} ###")
        print("#"*80 + "\n")

        all_activations = {}
        for concept, prompts in filtered_prompts.items():
            if not prompts:
                print(f"No prompts for concept '{concept}' after filtering. Skipping.")
                continue
            all_activations[concept] = get_word_token_activations(model, tokenizer, prompts, target_layer, concept)
            gc.collect()
            torch.cuda.empty_cache()

        analysis_results = analyze_manifolds(all_activations)

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

        # Find the token index for the test concept in the formatted prompt
        input_ids = inputs_for_gen[0].tolist()
        concept_ids = tokenizer.encode(test_concept, add_special_tokens=False)
        target_token_idx = -1
        # Find the last occurrence of the concept word
        for i in range(len(input_ids) - len(concept_ids) + 1):
            if input_ids[i:i+len(concept_ids)] == concept_ids:
                target_token_idx = i + len(concept_ids) - 1
        
        if target_token_idx == -1:
            print(f"Warning: Concept '{test_concept}' not found in the tokenized test prompt. Skipping perturbation tests for layer {target_layer}.")
            continue

        perturbation_scales = [-20.0, -10.0, -5.0, -2.5, -1.5, 0.0, 1.5, 2.5, 5.0, 10.0, 20.0]

        if test_concept not in analysis_results:
            print(f"No analysis results for concept '{test_concept}', skipping layer {target_layer}.")
            continue
        
        concept_analysis = analysis_results[test_concept]

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

            original_output = generate_with_perturbation(model, tokenizer, target_layer, pc_direction, 0, pc_eigenvalue, target_token_idx, inputs_for_gen)
            print(f"Original model completion: {original_output}")
            
            perturbation_scales = [-20.0, -10.0, -5.0, -2.5, -1.5, 0.0, 1.5, 2.5, 5.0, 10.0, 20.0]
            
            print(f"\n--- Perturbing token activation for '{test_concept}' along PC{axis} ---")
            for scale in perturbation_scales:
                perturbed_output = generate_with_perturbation(
                    model, tokenizer, target_layer, pc_direction, scale, pc_eigenvalue, target_token_idx, inputs_for_gen
                )
                print(f"Perturbation scale {scale:+.1f}x: {perturbed_output}")

            print("\n" + "="*80)
            print(f"--- Analyzing original dataset prompts along the PC{axis} '{test_concept}' direction (Layer {target_layer}) ---")
            top_prompts = find_top_prompts(
                filtered_prompts[test_concept],
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

        if test_concept in analysis_results:
            concept_analysis = analysis_results[test_concept]
            effective_mask = concept_analysis["effective_mask"]
            
            # Find the first principal component that is *not* effective
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

                original_output = generate_with_perturbation(model, tokenizer, target_layer, ortho_direction, 0, ortho_eigenvalue, target_token_idx, inputs_for_gen)
                print(f"Original model completion: {original_output}")

                for scale in perturbation_scales:
                    perturbed_output = generate_with_perturbation(
                        model, tokenizer, target_layer, ortho_direction, scale, ortho_eigenvalue, target_token_idx, inputs_for_gen
                    )
                    print(f"Perturbation scale {scale:+.1f}x: {perturbed_output}")
            else:
                print("Could not find an orthogonal (ineffective) direction to perturb.")
        else:
            print(f"No analysis results for concept '{test_concept}', skipping orthogonal perturbation.")


    # --- FINAL PLOTTING ---
    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "wordToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "wordToken")

if __name__ == "__main__":
    main()
