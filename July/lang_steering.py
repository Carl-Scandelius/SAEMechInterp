#!/usr/bin/env python3
"""
Language Manifold Analysis and Cross-Language Steering.

This script examines how language model activations form manifolds when translating to
different target languages. We split dog-related prompts into two groups, one for Spanish
translation and one for German translation. The manifolds are analysed for similarities,
and we test whether perturbation in the direction of one language manifold can "steer"
the model toward that language.
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

def analyse_manifold_relationships(spanish_analysis, german_analysis, model_name_str):
    """Analyse relationships between Spanish and German translation manifolds."""
    print("\n" + "#"*80)
    print("### ANALYZING MANIFOLD RELATIONSHIPS ###")
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
    plt.title(f"Cosine Similarity Between Spanish and German PCs\n{model_name_str}")
    plt.tight_layout()
    plt.savefig(f"{model_name_str}_spanish_german_pc_similarity.png")
    plt.close()
    
    print("\nEigenvector similarity matrix saved as '{model_name_str}_spanish_german_pc_similarity.png'")
    
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
    print(f"\nConfiguration: PERTURB_ONCE is set to {PERTURB_ONCE}\n")
    print(f"Configuration: USE_SYSTEM_PROMPT_FOR_MANIFOLD is set to {USE_SYSTEM_PROMPT_FOR_MANIFOLD}\n")

    # Load and initialize model
    model_name_str = MODEL_NAME.split('/')[-1]
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)

    # Load prompts and split dog prompts randomly into two equal groups
    with open('prompts.json', 'r', encoding='utf-8') as f:
        concept_prompts = json.load(f)
    
    # Get all dog prompts and shuffle them
    dog_prompts = concept_prompts["dog"].copy()
    random.shuffle(dog_prompts)
    
    # Split into two equal groups
    split_point = len(dog_prompts) // 2
    spanish_prompts = dog_prompts[:split_point]
    german_prompts = dog_prompts[split_point:]
    
    print(f"Split {len(dog_prompts)} dog prompts into {len(spanish_prompts)} Spanish prompts and {len(german_prompts)} German prompts")
    
    TARGET_LAYERS = [0, 15, 31]  # Llama-3.1-8B has 32 layers (0-31)
    AXES_TO_ANALYZE = range(5)
    
    # System prompts for translation
    spanish_system_prompt = "You are a language model assistant. Please translate the following text accurately from English into Spanish:"
    german_system_prompt = "You are a language model assistant. Please translate the following text accurately from English into German:"
    
    for target_layer in TARGET_LAYERS:
        print("\n" + "#"*80)
        print(f"### STARTING ANALYSIS FOR LAYER {target_layer} ###")
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
        
        # Analyze manifold relationships
        analyse_manifold_relationships(
            analysis_results["dog into spanish"], 
            analysis_results["dog into german"],
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
        # This will help us understand if perturbing toward the Spanish manifold affects translation
        run_perturbation_experiment(
            model, tokenizer, messages_to_test, target_layer, 
            analysis_results["dog into spanish"], "dog into spanish", AXES_TO_ANALYZE, 
            target_token_idx=None, perturb_once=PERTURB_ONCE, orthogonal_mode=False
        )
        
        # Also run an experiment perturbing along the centroid vector (from German to Spanish)
        print("\n" + "="*80)
        print(f"--- PERTURBATION EXPERIMENT: PERTURBING ALONG CENTROID VECTOR ---")
        print(f"--- LAYER: {target_layer} ---")
        print("="*80)
        
        # Prepare inputs
        inputs = tokenizer.apply_chat_template(
            messages_to_test,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        # Get centroids
        spanish_centroid = analysis_results["dog into spanish"]["centroid"]
        german_centroid = analysis_results["dog into german"]["centroid"]
        centroid_vector = spanish_centroid - german_centroid
        
        # Find the eigenvalue to use for scaling the perturbation
        eigenvalue = analysis_results["dog into spanish"]["eigenvalues"][0]  # Use largest eigenvalue for scaling
        
        # Interpolate between German and Spanish centroids in 20% steps
        print("\n" + "="*80)
        print(f"--- INTERPOLATION BETWEEN GERMAN AND SPANISH CENTROIDS ---")
        print(f"--- LAYER: {target_layer} ---")
        print("="*80)
        
        # Format for display
        system_prompt = messages_to_test[0]['content'] if messages_to_test[0]['role'] == 'system' else ""
        user_prompt = next((msg['content'] for msg in messages_to_test if msg['role'] == 'user'), "")
        
        print(f"\nSystem Prompt: '{system_prompt}'")
        print(f"User Prompt:   '{user_prompt}'")
        
        # Prepare inputs
        inputs = tokenizer.apply_chat_template(
            messages_to_test,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        
        # Get original output without perturbation
        with torch.no_grad():
            output_ids = model.generate(
                inputs, max_new_tokens=50, temperature=0.7, top_p=0.9,
                do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id
            )
            prompt_length = inputs.shape[1]
            original_text = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
            
        print(f"\nOriginal output (German): {original_text}")
        
        # Apply interpolation in 20% steps from German (0%) to Spanish (100%)
        interpolation_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for step in interpolation_steps:
            # Skip 0.0 as it's the original German
            if step == 0.0:
                continue
                
            # Calculate the interpolated vector
            # At step=0.0, we're at German; at step=1.0, we're at Spanish
            perturbation = step * centroid_vector
            
            # Apply the perturbation
            perturbed_output = generate_with_perturbation(
                model, tokenizer, inputs, target_layer,
                perturbation / torch.norm(perturbation),  # Normalize direction vector
                1.0,  # Use constant scale since we're explicitly controlling magnitude via step
                eigenvalue,  # Use eigenvalue for appropriate scaling
                None,  # Perturb last token
                PERTURB_ONCE
            )
            
            print(f"\n{int(step * 100)}% toward Spanish: {perturbed_output}")
        
    print("\n" + "#"*80)
    print("### ANALYSIS COMPLETE ###")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
