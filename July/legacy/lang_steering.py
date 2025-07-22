#!/usr/bin/env python3
"""Cross-language manifold analysis and steering experiments. Probably defunct."""

from __future__ import annotations

import torch
from tqdm import tqdm
import gc
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
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

def get_final_token_activations(
    model, tokenizer, prompts, layer_idx, system_prompt=""
):
    """Extract final token activations from specified layer."""
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

def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two vectors."""
    dot_product = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    return dot_product / (norm1 * norm2)

def run_centroid_interpolation(
    model, tokenizer, messages_to_test: List[Dict[str, str]], 
    target_layer: int, spanish_analysis: Dict[str, Any], german_analysis: Dict[str, Any]
) -> None:
    """Run interpolation between German and Spanish translation centroids."""
    inputs = tokenizer.apply_chat_template(
        messages_to_test,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    
    spanish_centroid = spanish_analysis["centroid"].to(model.device)
    german_centroid = german_analysis["centroid"].to(model.device)
    
    centroid_vector = spanish_centroid - german_centroid
    centroid_distance = torch.norm(centroid_vector).item()
    direction_vector = centroid_vector / centroid_distance
    
    print("\n" + "="*80)
    print("--- INTERPOLATING FROM GERMAN TO SPANISH TRANSLATION ---")
    print("--- LAYER: {} ---".format(target_layer))
    print("--- CENTROID DISTANCE: {:.4f} ---".format(centroid_distance))
    print("="*80)
    
    system_prompt = messages_to_test[0]['content'] if messages_to_test[0]['role'] == 'system' else ""
    user_prompt = next((msg['content'] for msg in messages_to_test if msg['role'] == 'user'), "")
    
    print("\nSystem Prompt: '{}'".format(system_prompt))
    print("User Prompt:   '{}'".format(user_prompt))
    print("\nInterpolation Results:")
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs, max_new_tokens=50,
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id
        )
        prompt_length = inputs.shape[1]
        original_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
    
    print("\n0% (German baseline): {}".format(original_text))
    
    # Interpolate in 20% steps
    for step in range(1, 6):
        percentage = step * 20
        
        if step == 5:
            print("\n{}% (Spanish target):".format(percentage))
        else:
            print("\n{}% interpolation:".format(percentage))
        
        perturbation_magnitude = step * 0.2 * centroid_distance
        
        perturbed_output = generate_with_perturbation(
            model, tokenizer, inputs, target_layer,
            direction_vector,
            perturbation_magnitude,
            torch.tensor(1.0, device=model.device),
            None,
            PERTURB_ONCE
        )
        
        print(perturbed_output)

def plot_cross_layer_pc_similarity(
    all_layer_results: Dict[int, Dict[str, Any]], 
    concept: str, 
    num_pcs: int = 5, 
    model_name_str: str = ""
) -> None:
    """Plot PC similarity matrices across layers for a single concept."""
    layers = sorted(all_layer_results.keys())
    n_layers = len(layers)
    
    fig, axes = plt.subplots(n_layers, n_layers, figsize=(20, 20))
    fig.suptitle("Cross-Layer PC Similarity for '{}' Concept\n{}".format(concept, model_name_str), fontsize=16)
    
    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            eigenvectors_i = all_layer_results[layer_i][concept]["eigenvectors"][:num_pcs]
            eigenvectors_j = all_layer_results[layer_j][concept]["eigenvectors"][:num_pcs]
            
            similarity_matrix = torch.zeros((num_pcs, num_pcs))
            for pi in range(num_pcs):
                for pj in range(num_pcs):
                    if pi < len(eigenvectors_i) and pj < len(eigenvectors_j):
                        similarity = compute_cosine_similarity(eigenvectors_i[pi], eigenvectors_j[pj])
                        similarity_matrix[pi, pj] = similarity
            
            im = sns.heatmap(similarity_matrix.cpu().numpy(), 
                       annot=True, fmt='.2', cmap='coolwarm', 
                       vmin=-1, vmax=1, center=0, 
                       xticklabels=["PC{}".format(i) for i in range(num_pcs)],
                       yticklabels=["PC{}".format(i) for i in range(num_pcs)],
                       ax=axes[i, j], cbar=False)
            
            if i == 0:
                axes[i, j].set_title("Layer {}".format(layer_j))
            if j == 0:
                axes[i, j].set_ylabel("Layer {}".format(layer_i))
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = "{}_{}_cross_layer_pc_similarity.png".format(model_name_str, concept.replace(' ', '_'))
    plt.savefig(filename)
    plt.close()
    print("Saved cross-layer PC similarity matrix for '{}' to {}".format(concept, filename))

def plot_cross_concept_pc_similarity_by_layer(
    all_layer_results: Dict[int, Dict[str, Any]], 
    concepts: List[str], 
    layer: int, 
    num_pcs: int = 5, 
    model_name_str: str = ""
) -> None:
    """Plot PC similarity matrix between concepts at a specific layer."""
    if layer not in all_layer_results:
        print("Warning: No data for layer {}".format(layer))
        return
        
    layer_data = all_layer_results[layer]
    plt.figure(figsize=(10, 8))
    
    similarity_matrix = torch.zeros((len(concepts) * num_pcs, len(concepts) * num_pcs))
    xlabels = []
    ylabels = []
    
    for i, concept_i in enumerate(concepts):
        if concept_i not in layer_data:
            continue
            
        eigenvectors_i = layer_data[concept_i]["eigenvectors"][:num_pcs]
        for pi in range(min(num_pcs, len(eigenvectors_i))):
            ylabels.append("{}\nPC{}".format(concept_i, pi))
            
            for j, concept_j in enumerate(concepts):
                if concept_j not in layer_data:
                    continue
                    
                eigenvectors_j = layer_data[concept_j]["eigenvectors"][:num_pcs]
                for pj in range(min(num_pcs, len(eigenvectors_j))):
                    if i == 0:
                        xlabels.append("{}\nPC{}".format(concept_j, pj))
                    
                    sim = compute_cosine_similarity(eigenvectors_i[pi], eigenvectors_j[pj])
                    similarity_matrix[i * num_pcs + pi, j * num_pcs + pj] = sim
    
    sns.heatmap(similarity_matrix[:len(ylabels), :len(xlabels)].cpu().numpy(), 
                annot=True, fmt='.2', cmap='coolwarm',
                vmin=-1, vmax=1, center=0,
                xticklabels=xlabels, yticklabels=ylabels)
    
    plt.title("Cross-Concept PC Similarity at Layer {}\n{}".format(layer, model_name_str))
    plt.tight_layout()
    filename = "{}_layer_{}_cross_concept_pc_similarity.png".format(model_name_str, layer)
    plt.savefig(filename)
    plt.close()
    print("Saved cross-concept PC similarity matrix for layer {} to {}".format(layer, filename))

def plot_centroid_distances_across_layers(
    all_layer_results: Dict[int, Dict[str, Any]], 
    concepts: List[str], 
    model_name_str: str = ""
) -> None:
    """Plot centroid distances between concepts across layers."""
    layers = sorted(all_layer_results.keys())
    
    distances = {}
    for pair in [(concepts[i], concepts[j]) for i in range(len(concepts)) for j in range(i+1, len(concepts))]:
        distances["{}-{}".format(pair[0], pair[1])] = []
        
        for layer in layers:
            if pair[0] not in all_layer_results[layer] or pair[1] not in all_layer_results[layer]:
                distances["{}-{}".format(pair[0], pair[1])].append(float('nan'))
                continue
                
            centroid_1 = all_layer_results[layer][pair[0]]["centroid"]
            centroid_2 = all_layer_results[layer][pair[1]]["centroid"]
            distance = torch.norm(centroid_1 - centroid_2).item()
            distances["{}-{}".format(pair[0], pair[1])].append(distance)
    
    plt.figure(figsize=(12, 6))
    for pair_name, dists in distances.items():
        plt.plot(layers, dists, marker='o', label=pair_name)
        
    plt.xlabel('Layer')
    plt.ylabel('Centroid Distance')
    plt.title("Centroid Distances Across Layers\n{}".format(model_name_str))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(layers)
    
    filename = "{}_centroid_distances_across_layers.png".format(model_name_str)
    plt.savefig(filename)
    plt.close()
    print("Saved centroid distances plot to {}".format(filename))

def analyse_manifold_relationships(
    spanish_analysis: Dict[str, Any], 
    german_analysis: Dict[str, Any], 
    layer_idx: int, 
    model_name_str: str = ""
) -> None:
    """Analyze relationships between Spanish and German translation manifolds."""
    print("\n" + "#"*80)
    print("--- MANIFOLD RELATIONSHIP ANALYSIS (LAYER {}) ---".format(layer_idx))
    print("#"*80 + "\n")
    
    spanish_centroid = spanish_analysis["centroid"]
    german_centroid = german_analysis["centroid"]
    centroid_vector = spanish_centroid - german_centroid
    centroid_distance = torch.norm(centroid_vector).item()
    
    print("Distance between manifold centroids: {:.4f}".format(centroid_distance))
    
    normalized_centroid_vector = centroid_vector / torch.norm(centroid_vector)
    
    print("\nCosine similarity between centroids vector and Spanish PCs:")
    for i in range(min(5, len(spanish_analysis["eigenvectors"]))):
        pc_vector = spanish_analysis["eigenvectors"][i]
        similarity = compute_cosine_similarity(normalized_centroid_vector, pc_vector)
        print("PC{}: {:.4f}".format(i, similarity.item()))
    
    print("\nCosine similarity between centroids vector and German PCs:")
    for i in range(min(5, len(german_analysis["eigenvectors"]))):
        pc_vector = german_analysis["eigenvectors"][i]
        similarity = compute_cosine_similarity(normalized_centroid_vector, pc_vector)
        print("PC{}: {:.4f}".format(i, similarity.item()))
    
    num_components = min(10, min(len(spanish_analysis["eigenvectors"]), len(german_analysis["eigenvectors"])))
    similarity_matrix = torch.zeros((num_components, num_components))
    
    for i in range(num_components):
        for j in range(num_components):
            spanish_pc = spanish_analysis["eigenvectors"][i]
            german_pc = german_analysis["eigenvectors"][j]
            similarity = compute_cosine_similarity(spanish_pc, german_pc)
            similarity_matrix[i, j] = similarity
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.cpu().numpy(), annot=True, fmt='.2', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0,
                xticklabels=["German PC{}".format(i) for i in range(num_components)],
                yticklabels=["Spanish PC{}".format(i) for i in range(num_components)])
    plt.title("Cosine Similarity Between Spanish and German PCs (Layer {})\n{}".format(layer_idx, model_name_str))
    plt.tight_layout()
    plt.savefig("{}_layer{}_spanish_german_pc_similarity.png".format(model_name_str, layer_idx))
    plt.close()
    
    print("\nEigenvector similarity matrix saved as '{}_layer{}_spanish_german_pc_similarity.png'".format(model_name_str, layer_idx))
    
    print("\nCosine similarity between corresponding Spanish and German PCs:")
    for i in range(min(5, min(len(spanish_analysis["eigenvectors"]), len(german_analysis["eigenvectors"])))):
        spanish_pc = spanish_analysis["eigenvectors"][i]
        german_pc = german_analysis["eigenvectors"][i]
        similarity = compute_cosine_similarity(spanish_pc, german_pc)
        print("PC{}: {:.4f}".format(i, similarity.item()))
    
    print("\nVariance explained by Spanish PCs:")
    spanish_total_var = spanish_analysis["eigenvalues"].sum().item()
    for i in range(min(5, len(spanish_analysis["eigenvalues"]))):
        variance_explained = spanish_analysis["eigenvalues"][i].item() / spanish_total_var * 100
        print("PC{}: {:.2f}%".format(i, variance_explained))
    
    print("\nVariance explained by German PCs:")
    german_total_var = german_analysis["eigenvalues"].sum().item()
    for i in range(min(5, len(german_analysis["eigenvalues"]))):
        variance_explained = german_analysis["eigenvalues"][i].item() / german_total_var * 100
        print("PC{}: {:.2f}%".format(i, variance_explained))

def main() -> None:
    """Run language manifold analysis and perturbation experiments."""
    with open("prompts.json", "r") as f:
        concept_prompts = json.load(f)
    
    model_name_str = MODEL_NAME.split("/")[-1]
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    
    print("Using model: {}".format(MODEL_NAME))
    
    dog_prompts = concept_prompts["dog"].copy()
    random.shuffle(dog_prompts)
    
    split_point = len(dog_prompts) // 2
    spanish_prompts = dog_prompts[:split_point]
    german_prompts = dog_prompts[split_point:]
    
    print("Split {} dog prompts into {} Spanish prompts and {} German prompts".format(len(dog_prompts), len(spanish_prompts), len(german_prompts)))
    
    TARGET_LAYERS = [0, 15, 31]
    ANALYSIS_LAYERS = [0, 15, 31]
        # Multi-axis analysis removed - feature disabled
    
    spanish_system_prompt = "You are a language model assistant. Please translate the following text accurately from English into Spanish:"
    german_system_prompt = "You are a language model assistant. Please translate the following text accurately from English into German:"
    
    all_layer_results = {}
    
    for target_layer in TARGET_LAYERS:
        print("\n" + "#"*80)
        print("### COLLECTING DATA FOR LAYER {} ###".format(target_layer))
        print("#"*80 + "\n")
        
        all_activations = {}
        
        print("\nGathering activations for dog into Spanish...")
        system_prompt_for_manifold = spanish_system_prompt if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
        all_activations["dog into spanish"] = get_final_token_activations(
            model, tokenizer, spanish_prompts, target_layer, 
            system_prompt=system_prompt_for_manifold
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\nGathering activations for dog into German...")
        system_prompt_for_manifold = german_system_prompt if USE_SYSTEM_PROMPT_FOR_MANIFOLD else ""
        all_activations["dog into german"] = get_final_token_activations(
            model, tokenizer, german_prompts, target_layer, 
            system_prompt=system_prompt_for_manifold
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\nAnalyzing manifolds...")
        analysis_results = analyse_manifolds(all_activations, local_centre=False)  # Default to global centering
        all_layer_results[target_layer] = analysis_results
        
        if target_layer in ANALYSIS_LAYERS:
            analyse_manifold_relationships(
                analysis_results["dog into spanish"], 
                analysis_results["dog into german"],
                target_layer,
                model_name_str
            )
            
            test_prompt = "The dog ran around the park. It was a labrador."
            
            messages_to_test = [
                {"role": "system", "content": german_system_prompt},
                {"role": "user", "content": test_prompt}
            ]
            
            print("\n" + "="*80)
            print("--- PERTURBATION EXPERIMENT: GERMAN TRANSLATION PERTURBED TOWARD SPANISH ---")
            print("--- LAYER: {} ---".format(target_layer))
            print("="*80)
            
            run_perturbation_experiment(
                model, tokenizer, messages_to_test, target_layer, 
                            analysis_results["dog into spanish"], "dog into spanish", 
                target_token_idx=None, perturb_once=PERTURB_ONCE, orthogonal_mode=False
            )
            
            run_centroid_interpolation(
                model, tokenizer, messages_to_test, target_layer, 
                analysis_results["dog into spanish"], 
                analysis_results["dog into german"]
            )
    
    print("\n" + "#"*80)
    print("### CROSS-LAYER ANALYSES ###")
    print("#"*80 + "\n")
    
    print("\nPlotting centroid distances across all layers...")
    plot_centroid_distances_across_layers(
        all_layer_results, 
        concepts=["dog into spanish", "dog into german"], 
        model_name_str=model_name_str
    )
    
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
    
    for layer in ANALYSIS_LAYERS:
        print("\nPlotting cross-concept PC similarity for layer {}...".format(layer))
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
