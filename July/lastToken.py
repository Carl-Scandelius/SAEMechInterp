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

def get_final_token_activations(model, tokenizer, prompts, layer_idx):
    activations = []

    def hook_fn(module, input, output):
        last_token_activation = output[0][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    print(f"Extracting activations from layer {layer_idx}...")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(DEVICE)
        with torch.no_grad():
            model(input_ids=inputs)

    hook_handle.remove()
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
        pca = PCA()
        pca.fit(acts.numpy())
        
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        mean_eigval = eigenvalues.mean()
        std_eigval = eigenvalues.std()
        threshold = mean_eigval #- 1.0 * std_eigval
        
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
    PERTURB_ONCE = False # true entails perturbing only the final token in the user prompt; false entails perturbing the final token in every subsequent pass through the model
    print(f"\nConfiguration: PERTURB_ONCE is set to {PERTURB_ONCE}\n")

    # Data for plotting across layers
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
        for concept, prompts in concept_prompts.items():
            all_activations[concept] = get_final_token_activations(model, tokenizer, prompts, target_layer)
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

            original_output = generate_with_perturbation(model, tokenizer, messages_to_test, target_layer, ortho_direction, 0, ortho_eigenvalue, perturb_once=PERTURB_ONCE)
            print(f"Original model completion: {original_output}")

            for scale in perturbation_scales:
                perturbed_output = generate_with_perturbation(
                    model, tokenizer, messages_to_test, target_layer, ortho_direction, scale, ortho_eigenvalue,
                    perturb_once=PERTURB_ONCE
                )
                print(f"Perturbation scale {scale:+.1f}x: {perturbed_output}")
        else:
            print("Could not find an orthogonal (ineffective) direction to perturb.")

    # --- FINAL PLOTTING ---
    print("\n" + "#"*80)
    print("### PLOTTING OVERALL RESULTS ###")
    print("#"*80 + "\n")
    plot_avg_eigenvalues(dog_avg_eigenvalues, model_name_str, "lastToken")
    plot_similarity_matrix(dog_top_eigenvectors, model_name_str, "lastToken")

if __name__ == "__main__":
    main()