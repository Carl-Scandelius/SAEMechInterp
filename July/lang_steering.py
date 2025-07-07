#!/usr/bin/env python3
"""
Language Manifold Analysis and Cross-Language Steering.

This script examines how language model activations form manifolds when translating to
different target languages. We split dog-related prompts into two groups, one for Spanish
translation and one for German translation. The manifolds are analysed for similarities,
and we test whether perturbation in the direction of one language manifold can "steer"
the model toward that language.

Advanced analysis features:
- Cross-language PC similarity matrices at each layer
- Within-language PC similarity matrices across layers
- Centroid distance analysis across all layers
- Interpolation between language manifolds
"""

import torch
from tqdm import tqdm
import gc
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import (
    get_model_and_tokenizer,
    analyse_manifolds,
    run_perturbation_experiment,
    generate_with_perturbation,
    MODEL_NAME,
    DEVICE
)
from transformers import logging
logging.set_verbosity(logging.ERROR)

USE_SYSTEM_PROMPT_FOR_MANIFOLD = True
PERTURB_ONCE = True

def get_final_token_activations(model, tokenizer, prompts, layer_idx, system_prompt=""):
    """Get activations of the final token in the response for each prompt."""
    activations = []

    def hook_fn(module, input, output):
        last_token_activation = output[0][:, -1, :].detach().cpu()
        activations.append(last_token_activation)

    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    for prompt in tqdm(prompts, desc="Processing prompts"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ] if system_prompt else [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            model(inputs)

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    hook_handle.remove()
    return torch.cat(activations, dim=0)

def compute_cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    dot_product = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    return dot_product / (norm1 * norm2)

def run_centroid_interpolation(model, tokenizer, messages_to_test, target_layer, spanish_analysis, german_analysis):
    """Run interpolation between German and Spanish translation centroids.
    
    Interpolates along the vector connecting the German centroid to the Spanish centroid
    in 20% steps of the total distance between them.
    At 0% we are at the German centroid, at 100% we are at the Spanish centroid.
    """
    # Prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages_to_test,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    
    # Get centroids and move them to the model's device
    spanish_centroid = spanish_analysis["centroid"].to(model.device)
    german_centroid = german_analysis["centroid"].to(model.device)
    
    # Vector from German to Spanish centroid
    centroid_vector = spanish_centroid - german_centroid
    
    # Compute the total distance between centroids
    centroid_distance = torch.norm(centroid_vector).item()
    
    # Normalize the direction vector
    direction_vector = centroid_vector / centroid_distance
    
    print("\n" + "="*80)
    print(f"--- INTERPOLATING FROM GERMAN TO SPANISH TRANSLATION ---")
    print(f"--- LAYER: {target_layer} ---")
    print(f"--- CENTROID DISTANCE: {centroid_distance:.4f} ---")
    print("="*80)
    
    # Format for display
    system_prompt = messages_to_test[0]['content'] if messages_to_test[0]['role'] == 'system' else ""
    user_prompt = next((msg['content'] for msg in messages_to_test if msg['role'] == 'user'), "")
    
    print(f"\nSystem Prompt: '{system_prompt}'")
    print(f"User Prompt:   '{user_prompt}'")
    print("\nInterpolation Results:")
    
    # Get the original (baseline) output without any perturbation
    with torch.no_grad():
        output_ids = model.generate(
            inputs, max_new_tokens=50, temperature=0.7, top_p=0.9,
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id
        )
        prompt_length = inputs.shape[1]
        original_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    
    print(f"\n0% (German baseline): {original_text}")
    
    # Interpolate in 20% steps (5 steps from 20% to 100%)
    for step in range(1, 6):  # 20%, 40%, 60%, 80%, 100%
        percentage = step * 20
        
        if step == 5:
            print(f"\n{percentage}% (Spanish target):")
        else:
            print(f"\n{percentage}% interpolation:")
        
        # Calculate perturbation magnitude for this step (as a fraction of total distance)
        # This ensures we're moving by equal *distance* steps, not just equal vector scaling
        perturbation_magnitude = step * 0.2 * centroid_distance
        
        # Generate with perturbation
        # Pass eigenvalue as tensor of 1.0 since we're directly controlling magnitude
        # and don't want additional eigenvalue scaling
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, target_layer,
            direction_vector,  # Unit vector in the direction from German to Spanish
            perturbation_magnitude,  # Magnitude is the fraction of total distance
            torch.tensor(1.0, device=model.device),  # Eigenvalue of 1.0 as tensor for neutral scaling
            None,  # Perturb last token
            PERTURB_ONCE
        )
        
        print(perturbed_output)

def plot_cross_layer_pc_similarity(all_layer_results, concept, num_pcs=5, model_name_str=""):
    """Plot PC similarity matrices across different layers for a single concept."""
    layers = sorted(all_layer_results.keys())
    n_layers = len(layers)
    
    # Create a figure with n_layers × n_layers subplots
    fig, axes = plt.subplots(n_layers, n_layers, figsize=(20, 20))
    fig.suptitle(f"Cross-Layer PC Similarity for '{concept}' Concept\n{model_name_str}", fontsize=16)
    
    # For each pair of layers, calculate PC similarity
    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            # Get eigenvectors from both layers
            eigenvectors_i = all_layer_results[layer_i][concept]["eigenvectors"][:num_pcs]
            eigenvectors_j = all_layer_results[layer_j][concept]["eigenvectors"][:num_pcs]
            
            # Calculate similarity matrix
            similarity_matrix = torch.zeros((num_pcs, num_pcs))
            for pi in range(num_pcs):
                for pj in range(num_pcs):
                    if pi < len(eigenvectors_i) and pj < len(eigenvectors_j):  # Check bounds
                        similarity = compute_cosine_similarity(eigenvectors_i[pi], eigenvectors_j[pj])
                        similarity_matrix[pi, pj] = similarity
            
            # Plot heatmap
            im = sns.heatmap(similarity_matrix.cpu().numpy(), 
                       annot=True, fmt='.2f', cmap='coolwarm', 
                       vmin=-1, vmax=1, center=0, 
                       xticklabels=[f"PC{i}" for i in range(num_pcs)],
                       yticklabels=[f"PC{i}" for i in range(num_pcs)],
                       ax=axes[i, j], cbar=False)
            
            if i == 0:
                axes[i, j].set_title(f"Layer {layer_j}")
            if j == 0:
                axes[i, j].set_ylabel(f"Layer {layer_i}")
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"{model_name_str}_{concept.replace(' ', '_')}_cross_layer_pc_similarity.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved cross-layer PC similarity matrix for '{concept}' to {filename}")

def plot_cross_concept_pc_similarity_by_layer(all_layer_results, concepts, layer, num_pcs=5, model_name_str=""):
    """Plot PC similarity matrix between concepts at a specific layer."""
    # Ensure we have data for the requested layer
    if layer not in all_layer_results:
        print(f"Warning: No data for layer {layer}")
        return
        
    # Get eigenvectors for each concept at this layer
    layer_data = all_layer_results[layer]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Calculate similarity matrix
    similarity_matrix = torch.zeros((len(concepts) * num_pcs, len(concepts) * num_pcs))
    
    # Labels for the axes
    xlabels = []
    ylabels = []
    
    # Fill the similarity matrix
    for i, concept_i in enumerate(concepts):
        if concept_i not in layer_data:
            continue
            
        eigenvectors_i = layer_data[concept_i]["eigenvectors"][:num_pcs]
        for pi in range(min(num_pcs, len(eigenvectors_i))):
            ylabels.append(f"{concept_i}\nPC{pi}")
            
            for j, concept_j in enumerate(concepts):
                if concept_j not in layer_data:
                    continue
                    
                eigenvectors_j = layer_data[concept_j]["eigenvectors"][:num_pcs]
                for pj in range(min(num_pcs, len(eigenvectors_j))):
                    if i == 0:  # Only add labels once
                        xlabels.append(f"{concept_j}\nPC{pj}")
                    
                    # Calculate similarity
                    sim = compute_cosine_similarity(eigenvectors_i[pi], eigenvectors_j[pj])
                    similarity_matrix[i * num_pcs + pi, j * num_pcs + pj] = sim
    
    # Plot heatmap
    sns.heatmap(similarity_matrix[:len(ylabels), :len(xlabels)].cpu().numpy(), 
                annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0,
                xticklabels=xlabels, yticklabels=ylabels)
    
    plt.title(f"Cross-Concept PC Similarity at Layer {layer}\n{model_name_str}")
    plt.tight_layout()
    filename = f"{model_name_str}_layer_{layer}_cross_concept_pc_similarity.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved cross-concept PC similarity matrix for layer {layer} to {filename}")

def plot_centroid_distances_across_layers(all_layer_results, concepts, model_name_str=""):
    """Plot centroid distances between concepts across all analyzed layers."""
    layers = sorted(all_layer_results.keys())
    
    # Calculate distances between each pair of concepts at each layer
    distances = {}
    for pair in [(concepts[i], concepts[j]) for i in range(len(concepts)) for j in range(i+1, len(concepts))]:
        distances[f"{pair[0]}-{pair[1]}"] = []
        
        for layer in layers:
            if pair[0] not in all_layer_results[layer] or pair[1] not in all_layer_results[layer]:
                distances[f"{pair[0]}-{pair[1]}"].append(float('nan'))
                continue
                
            centroid_1 = all_layer_results[layer][pair[0]]["centroid"]
            centroid_2 = all_layer_results[layer][pair[1]]["centroid"]
            distance = torch.norm(centroid_1 - centroid_2).item()
            distances[f"{pair[0]}-{pair[1]}"].append(distance)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    for pair_name, dists in distances.items():
        plt.plot(layers, dists, marker='o', label=pair_name)
        
    plt.xlabel('Layer')
    plt.ylabel('Centroid Distance')
    plt.title(f"Centroid Distances Across Layers\n{model_name_str}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(layers)
    
    filename = f"{model_name_str}_centroid_distances_across_layers.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved centroid distances plot to {filename}")

def analyse_manifold_relationships(spanish_analysis, german_analysis, layer_idx, model_name_str=""):
    """Analyse relationships between Spanish and German translation manifolds."""
    print("\n" + "#"*80)
    print(f"--- MANIFOLD RELATIONSHIP ANALYSIS (LAYER {layer_idx}) ---")
    print("#"*80 + "\n")
    
    # Get centroids and calculate centroid vector (the vector from German to Spanish centroid)
    spanish_centroid = spanish_analysis["centroid"]
    german_centroid = german_analysis["centroid"]
    centroid_vector = spanish_centroid - german_centroid
    centroid_distance = torch.norm(centroid_vector).item()
    
    print(f"Distance between manifold centroids: {centroid_distance:.4f}")
    
    # Normalize centroid vector for cosine similarity calculations
    normalized_centroid_vector = centroid_vector / torch.norm(centroid_vector)
    
    # Calculate cosine similarity between centroid vector and each PC
    print("\nCosine similarity between centroids vector and Spanish PCs:")
    for i in range(min(5, len(spanish_analysis["eigenvectors"]))):
        pc_vector = spanish_analysis["eigenvectors"][i]
        similarity = compute_cosine_similarity(normalized_centroid_vector, pc_vector)
        print(f"PC{i}: {similarity.item():.4f}")
    
    print("\nCosine similarity between centroids vector and German PCs:")
    for i in range(min(5, len(german_analysis["eigenvectors"]))):
        pc_vector = german_analysis["eigenvectors"][i]
        similarity = compute_cosine_similarity(normalized_centroid_vector, pc_vector)
        print(f"PC{i}: {similarity.item():.4f}")
    
    # Create a similarity matrix between Spanish and German eigenvectors
    num_components = min(10, min(len(spanish_analysis["eigenvectors"]), len(german_analysis["eigenvectors"])))
    similarity_matrix = torch.zeros((num_components, num_components))
    
    # Calculate all pairwise cosine similarities between eigenvectors
    for i in range(num_components):
        for j in range(num_components):
            spanish_pc = spanish_analysis["eigenvectors"][i]
            german_pc = german_analysis["eigenvectors"][j]
            similarity = compute_cosine_similarity(spanish_pc, german_pc)
            similarity_matrix[i, j] = similarity
    
    # Generate a heatmap of the similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.cpu().numpy(), annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0,
                xticklabels=[f"German PC{i}" for i in range(num_components)],
                yticklabels=[f"Spanish PC{i}" for i in range(num_components)])
    plt.title(f"Cosine Similarity Between Spanish and German PCs (Layer {layer_idx})\n{model_name_str}")
    plt.tight_layout()
    plt.savefig(f"{model_name_str}_layer{layer_idx}_spanish_german_pc_similarity.png")
    plt.close()
    
    print(f"\nEigenvector similarity matrix saved as '{model_name_str}_layer{layer_idx}_spanish_german_pc_similarity.png'")
    
    # Calculate cosine similarity between corresponding PCs
    print("\nCosine similarity between corresponding Spanish and German PCs:")
    for i in range(min(5, min(len(spanish_analysis["eigenvectors"]), len(german_analysis["eigenvectors"])))):
        spanish_pc = spanish_analysis["eigenvectors"][i]
        german_pc = german_analysis["eigenvectors"][i]
        similarity = compute_cosine_similarity(spanish_pc, german_pc)
        print(f"PC{i}: {similarity.item():.4f}")
    
    # Compute variance explained by each PC for both languages
    print("\nVariance explained by Spanish PCs:")
    spanish_total_var = spanish_analysis["eigenvalues"].sum().item()
    for i in range(min(5, len(spanish_analysis["eigenvalues"]))):
        variance_explained = spanish_analysis["eigenvalues"][i].item() / spanish_total_var * 100
        print(f"PC{i}: {variance_explained:.2f}%")
    
    print("\nVariance explained by German PCs:")
    german_total_var = german_analysis["eigenvalues"].sum().item()
    for i in range(min(5, len(german_analysis["eigenvalues"]))):
        variance_explained = german_analysis["eigenvalues"][i].item() / german_total_var * 100
        print(f"PC{i}: {variance_explained:.2f}%")

def main():
    """Run language manifold analysis and perturbation experiments."""
    # Load prompts from JSON file
    with open("prompts.json", "r") as f:
        concept_prompts = json.load(f)
    
    # Load model and tokenizer
    model_name_str = MODEL_NAME.split("/")[-1]
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    
    print(f"Using model: {MODEL_NAME}")
    
    # Get all dog prompts and shuffle them
    dog_prompts = concept_prompts["dog"].copy()
    random.shuffle(dog_prompts)
    
    # Split into two equal groups
    split_point = len(dog_prompts) // 2
    spanish_prompts = dog_prompts[:split_point]
    german_prompts = dog_prompts[split_point:]
    
    print(f"Split {len(dog_prompts)} dog prompts into {len(spanish_prompts)} Spanish prompts and {len(german_prompts)} German prompts")
    
    # Only analyze specific layers of interest
    TARGET_LAYERS = [0, 15, 31]  # Only analyze these layers
    ANALYSIS_LAYERS = [0, 15, 31]  # Perform detailed analysis on all target layers
    AXES_TO_ANALYZE = range(5)
    
    # System prompts for translation
    spanish_system_prompt = "You are a language model assistant. Please translate the following text accurately from English into Spanish:"
    german_system_prompt = "You are a language model assistant. Please translate the following text accurately from English into German:"
    
    # Dictionary to store analysis results for each layer
    all_layer_results = {}
    
    # First pass: collect data from all layers
    for target_layer in TARGET_LAYERS:
        print("\n" + "#"*80)
        print(f"### COLLECTING DATA FOR LAYER {target_layer} ###")
        print("#"*80 + "\n")
        
        # Dictionary to store activations for different concepts
        all_activations = {}
        
        # Get activations for Spanish translation prompts
        print("\nGathering activations for dog into Spanish...")
        system_prompt_for_manifold = spanish_system_prompt if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
        all_activations["dog into spanish"] = get_final_token_activations(
            model, tokenizer, spanish_prompts, target_layer, 
            system_prompt=system_prompt_for_manifold
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get activations for German translation prompts
        print("\nGathering activations for dog into German...")
        system_prompt_for_manifold = german_system_prompt if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
        all_activations["dog into german"] = get_final_token_activations(
            model, tokenizer, german_prompts, target_layer, 
            system_prompt=system_prompt_for_manifold
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        # Analyze manifolds
        print("\nAnalyzing manifolds...")
        analysis_results = analyse_manifolds(all_activations)
        
        # Store analysis results for this layer
        all_layer_results[target_layer] = analysis_results
        
        # If this is one of the layers for detailed analysis, do perturbation experiments
        if target_layer in ANALYSIS_LAYERS:
            # Analyze manifold relationships
            analyse_manifold_relationships(
                analysis_results["dog into spanish"], 
                analysis_results["dog into german"],
                target_layer,
                model_name_str
            )
            
            # Test prompt for perturbation experiments
            test_prompt = "The dog ran around the park. It was a labrador."
            
            # Prepare messages for perturbation experiment - using German system prompt but perturbing toward Spanish
            messages_to_test = [
                {"role": "system", "content": german_system_prompt},
                {"role": "user", "content": test_prompt}
            ]
            
            # Run perturbation experiments in the direction of Spanish manifold
            print("\n" + "="*80)
            print(f"--- PERTURBATION EXPERIMENT: GERMAN TRANSLATION PERTURBED TOWARD SPANISH ---")
            print(f"--- LAYER: {target_layer} ---")
            print("="*80)
            
            # Run perturbation experiment along principal components of the Spanish manifold
            run_perturbation_experiment(
                model, tokenizer, messages_to_test, target_layer, 
                analysis_results["dog into spanish"], "dog into spanish", AXES_TO_ANALYZE, 
                target_token_idx=None, perturb_once=PERTURB_ONCE, orthogonal_mode=False
            )
            
            # Run centroid interpolation experiment
            run_centroid_interpolation(
                model, tokenizer, messages_to_test, target_layer, 
                analysis_results["dog into spanish"], 
                analysis_results["dog into german"]
            )
    
    # Second pass: cross-layer analyses using all collected data
    print("\n" + "#"*80)
    print(f"### CROSS-LAYER ANALYSES ###")
    print("#"*80 + "\n")
    
    # Plot centroid distances across all layers
    print("\nPlotting centroid distances across all layers...")
    plot_centroid_distances_across_layers(
        all_layer_results, 
        concepts=["dog into spanish", "dog into german"], 
        model_name_str=model_name_str
    )
    
    # Plot cross-layer PC similarity for each concept
    print("\nPlotting cross-layer PC similarity for Spanish translation...")
    plot_cross_layer_pc_similarity(
        all_layer_results, 
        concept="dog into spanish", 
        num_pcs=5, 
        model_name_str=model_name_str
    )
    
    print("\nPlotting cross-layer PC similarity for German translation...")
    plot_cross_layer_pc_similarity(
        all_layer_results, 
        concept="dog into german", 
        num_pcs=5, 
        model_name_str=model_name_str
    )
    
    # Plot cross-concept PC similarity for selected layers
    for layer in ANALYSIS_LAYERS:
        print(f"\nPlotting cross-concept PC similarity for layer {layer}...")
        plot_cross_concept_pc_similarity_by_layer(
            all_layer_results, 
            concepts=["dog into spanish", "dog into german"], 
            layer=layer, 
            num_pcs=5, 
            model_name_str=model_name_str
        )
    
    print("\n" + "#"*80)
    print("### ANALYSIS COMPLETE ###")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
