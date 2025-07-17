"""Word embedding analysis for variations and contextual development."""

from __future__ import annotations

import torch
from tqdm import tqdm
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from helpers import (
    get_model_and_tokenizer,
    analyse_manifolds,
    plot_avg_eigenvalues,
    plot_similarity_matrix,
    MODEL_NAME,
    DEVICE,
    ensure_tensor_compatibility,
)
from transformers import logging
logging.set_verbosity(40)

USE_SYSTEM_PROMPT_FOR_MANIFOLD = False 
SYSTEM_PROMPT = "What is the following sentence about:"

# Word variations to analyze
WORD_VARIATIONS = {
    "dog": ["dog", "dogs", "dog's", "dogs'", "puppy", "puppies", "puppy's", "puppies'"]
}

# Context levels for progressive analysis
CONTEXT_TEMPLATES = [
    "{word}.",  # No context - just the word
    "The {word}.",  # Simple determiner
    "The four-legged {word}.",
    "The barking {word}.",  # Basic predicate
    "The barking and bushy tailed {word}.",  # Extended sentence
    "The barking and bushy tailed, collared guide {word}.",  # Multi-sentence context
]


def extract_word_embedding(model, tokenizer, text: str, target_word: str, layer_idx: int, use_system_prompt: bool = True) -> Optional[torch.Tensor]:
    """Extract embedding for a specific word in given text."""
    
    # Prepare the input with optional system prompt
    if use_system_prompt and USE_SYSTEM_PROMPT_FOR_MANIFOLD:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Find the user content part in the full text to locate the target word
        user_content_start = full_text.lower().find(text.lower())
        if user_content_start == -1:
            print(f"Warning: User content not found in chat template output")
            # Fallback to direct text
            full_text = text
            word_start_offset = 0
        else:
            word_start_offset = user_content_start
    else:
        full_text = text
        word_start_offset = 0
    
    # Tokenize the full text
    inputs = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True).to(DEVICE)
    offset_mapping = inputs.pop('offset_mapping').squeeze(0)
    
    # Find target word in the original text
    text_lower = text.lower()
    word_lower = target_word.lower()
    local_word_start = text_lower.find(word_lower)
    
    if local_word_start == -1:
        print(f"Warning: Word '{target_word}' not found in text '{text}'")
        return None
    
    # Adjust for the full text with system prompt
    word_start = word_start_offset + local_word_start
    word_end = word_start + len(target_word)
    
    # Find corresponding token indices
    target_token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        if max(start, word_start) < min(end, word_end):
            target_token_indices.append(i)
    
    if not target_token_indices:
        print(f"Warning: No tokens found for word '{target_word}' in text '{text}'")
        return None
    
    # Use the last token of the word for consistency
    target_token_idx = max(target_token_indices)
    
    # Extract activation from specified layer
    activation_storage = []
    
    def hook_fn(module, input, output):
        # Extract the target token's activation
        token_activation = output[0][:, target_token_idx, :].detach().cpu()
        activation_storage.append(token_activation)
    
    hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        
        if activation_storage:
            return activation_storage[0].squeeze(0)
        else:
            return None
    finally:
        hook_handle.remove()


def analyze_single_word_variations(model, tokenizer, concept: str, variations: List[str], layers: List[int]) -> Dict[str, Any]:
    """Analyze embeddings for single-word variations across layers."""
    
    print(f"\n=== SINGLE-WORD VARIATION ANALYSIS: {concept.upper()} ===")
    
    results = {
        "concept": concept,
        "variations": variations,
        "layers": layers,
        "embeddings": {},  # layer -> variation -> embedding
        "similarities": {},  # layer -> similarity matrix
        "pca_results": {}  # layer -> PCA analysis
    }
    
    for layer_idx in layers:
        print(f"\nAnalyzing layer {layer_idx}...")
        
        layer_embeddings = {}
        
        for variation in variations:
            embedding = extract_word_embedding(model, tokenizer, variation, variation, layer_idx)
            if embedding is not None:
                layer_embeddings[variation] = embedding
            else:
                print(f"Failed to extract embedding for '{variation}' at layer {layer_idx}")
        
        if len(layer_embeddings) < 2:
            print(f"Insufficient embeddings for layer {layer_idx}, skipping analysis")
            continue
        
        results["embeddings"][layer_idx] = layer_embeddings
        
        # Compute pairwise cosine similarities
        variations_with_embeddings = list(layer_embeddings.keys())
        n_vars = len(variations_with_embeddings)
        similarity_matrix = torch.zeros(n_vars, n_vars)
        
        for i, var1 in enumerate(variations_with_embeddings):
            for j, var2 in enumerate(variations_with_embeddings):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    emb1 = layer_embeddings[var1]
                    emb2 = layer_embeddings[var2]
                    sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                    similarity_matrix[i, j] = sim
        
        results["similarities"][layer_idx] = {
            "matrix": similarity_matrix,
            "labels": variations_with_embeddings
        }
        
        # PCA analysis
        embeddings_tensor = torch.stack(list(layer_embeddings.values()))
        if embeddings_tensor.shape[0] > 1:
            # Compute PCA using existing helper
            activations_dict = {f"{concept}_{layer_idx}": embeddings_tensor}
            pca_analysis = analyse_manifolds(activations_dict, local_centre=False)
            results["pca_results"][layer_idx] = pca_analysis[f"{concept}_{layer_idx}"]
    
    return results


def analyze_contextual_development(model, tokenizer, concept: str, base_word: str, layers: List[int]) -> Dict[str, Any]:
    """Analyze how word embeddings develop with increasing context."""
    
    print(f"\n=== CONTEXTUAL DEVELOPMENT ANALYSIS: {base_word.upper()} ===")
    
    results = {
        "concept": concept,
        "base_word": base_word,
        "layers": layers,
        "context_templates": CONTEXT_TEMPLATES,
        "embeddings": {},  # layer -> context_level -> embedding
        "similarities": {},  # layer -> similarity analysis
        "context_progression": {}  # layer -> progression metrics
    }
    
    for layer_idx in layers:
        print(f"\nAnalyzing contextual development at layer {layer_idx}...")
        
        layer_embeddings = {}
        context_texts = []
        
        for i, template in enumerate(CONTEXT_TEMPLATES):
            context_text = template.format(word=base_word)
            context_texts.append(context_text)
            
            embedding = extract_word_embedding(model, tokenizer, context_text, base_word, layer_idx)
            if embedding is not None:
                layer_embeddings[f"context_{i}"] = {
                    "embedding": embedding,
                    "text": context_text,
                    "context_level": i
                }
            else:
                print(f"Failed to extract embedding for context level {i}: '{context_text}'")
        
        if len(layer_embeddings) < 2:
            print(f"Insufficient embeddings for layer {layer_idx}, skipping analysis")
            continue
        
        results["embeddings"][layer_idx] = layer_embeddings
        
        # Analyze progression: how does embedding change with more context?
        embeddings_list = [data["embedding"] for data in layer_embeddings.values()]
        n_contexts = len(embeddings_list)
        
        # Compute similarities to the base word (no context)
        base_embedding = embeddings_list[0]  # First is just the word alone
        context_similarities = []
        
        for i, emb in enumerate(embeddings_list):
            sim = torch.cosine_similarity(base_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
            context_similarities.append(sim)
        
        # Compute adjacent context similarities (how much does each step change things?)
        adjacent_similarities = []
        for i in range(1, len(embeddings_list)):
            sim = torch.cosine_similarity(
                embeddings_list[i-1].unsqueeze(0), 
                embeddings_list[i].unsqueeze(0)
            ).item()
            adjacent_similarities.append(sim)
        
        # Compute embedding distances/norms
        embedding_norms = [torch.norm(emb).item() for emb in embeddings_list]
        
        results["similarities"][layer_idx] = {
            "to_base": context_similarities,
            "adjacent": adjacent_similarities,
            "embedding_norms": embedding_norms
        }
        
        # Create full similarity matrix
        similarity_matrix = torch.zeros(n_contexts, n_contexts)
        for i in range(n_contexts):
            for j in range(n_contexts):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = torch.cosine_similarity(
                        embeddings_list[i].unsqueeze(0), 
                        embeddings_list[j].unsqueeze(0)
                    ).item()
                    similarity_matrix[i, j] = sim
        
        results["context_progression"][layer_idx] = {
            "similarity_matrix": similarity_matrix,
            "context_labels": [f"Level_{i}" for i in range(n_contexts)]
        }
    
    return results


def plot_word_variation_similarities(results: Dict[str, Any], save_prefix: str = "word_variations") -> None:
    """Plot similarity matrices for word variations across layers."""
    
    concept = results["concept"]
    layers = results["layers"]
    
    n_layers = len(layers)
    if n_layers == 0:
        return
    
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for i, layer_idx in enumerate(layers):
        if layer_idx not in results["similarities"]:
            continue
        
        sim_data = results["similarities"][layer_idx]
        matrix = sim_data["matrix"]
        labels = sim_data["labels"]
        
        sns.heatmap(
            matrix.numpy(), 
            annot=True, 
            fmt='.3f', 
            cmap='viridis',
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[i],
            vmin=0, vmax=1
        )
        axes[i].set_title(f'Layer {layer_idx}\n{concept.capitalize()} Variations')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    filename = f"{save_prefix}_{concept}_similarities.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close()


def plot_contextual_development(results: Dict[str, Any], save_prefix: str = "contextual_dev") -> None:
    """Plot contextual development analysis."""
    
    base_word = results["base_word"]
    layers = results["layers"]
    
    if not layers:
        return
    
    # Plot 1: Similarity to base word across context levels
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for layer_idx in layers:
        if layer_idx not in results["similarities"]:
            continue
        
        sim_data = results["similarities"][layer_idx]
        context_levels = list(range(len(sim_data["to_base"])))
        
        axes[0].plot(context_levels, sim_data["to_base"], 
                    marker='o', label=f'Layer {layer_idx}')
        
        if len(sim_data["adjacent"]) > 0:
            axes[1].plot(context_levels[1:], sim_data["adjacent"], 
                        marker='s', label=f'Layer {layer_idx}')
    
    axes[0].set_title(f'Similarity to Base Word "{base_word}" vs Context Level')
    axes[0].set_xlabel('Context Level')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Adjacent Context Similarity (Context Level i vs i+1)')
    axes[1].set_xlabel('Context Level Transition')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_prefix}_{base_word}_progression.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close()
    
    # Plot 2: Context similarity matrices
    n_layers = len([l for l in layers if l in results["context_progression"]])
    if n_layers > 0:
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
        if n_layers == 1:
            axes = [axes]
        
        plot_idx = 0
        for layer_idx in layers:
            if layer_idx not in results["context_progression"]:
                continue
            
            prog_data = results["context_progression"][layer_idx]
            matrix = prog_data["similarity_matrix"]
            labels = [f"L{i}" for i in range(len(prog_data["context_labels"]))]
            
            sns.heatmap(
                matrix.numpy(), 
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[plot_idx],
                vmin=0, vmax=1
            )
            axes[plot_idx].set_title(f'Layer {layer_idx}\nContext Similarities')
            plot_idx += 1
        
        plt.tight_layout()
        filename = f"{save_prefix}_{base_word}_context_matrix.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        plt.close()


def print_analysis_summary(single_word_results: Dict[str, Any], contextual_results: Dict[str, Any]) -> None:
    """Print summary of analysis results."""
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    # Single word variation summary
    print("\n--- SINGLE-WORD VARIATION ANALYSIS ---")
    for concept, results in single_word_results.items():
        print(f"\nConcept: {concept}")
        print(f"Variations analyzed: {', '.join(results['variations'])}")
        
        for layer_idx in results["layers"]:
            if layer_idx in results["similarities"]:
                sim_data = results["similarities"][layer_idx]
                matrix = sim_data["matrix"]
                labels = sim_data["labels"]
                
                # Find most and least similar pairs
                n = matrix.shape[0]
                max_sim = -1
                min_sim = 2
                max_pair = None
                min_pair = None
                
                for i in range(n):
                    for j in range(i+1, n):
                        sim_val = matrix[i, j].item()
                        if sim_val > max_sim:
                            max_sim = sim_val
                            max_pair = (labels[i], labels[j])
                        if sim_val < min_sim:
                            min_sim = sim_val
                            min_pair = (labels[i], labels[j])
                
                print(f"  Layer {layer_idx}:")
                if max_pair:
                    print(f"    Most similar: {max_pair[0]} ↔ {max_pair[1]} (sim: {max_sim:.3f})")
                if min_pair:
                    print(f"    Least similar: {min_pair[0]} ↔ {min_pair[1]} (sim: {min_sim:.3f})")
    
    # Contextual development summary
    print("\n--- CONTEXTUAL DEVELOPMENT ANALYSIS ---")
    for concept, results in contextual_results.items():
        print(f"\nConcept: {concept}")
        base_word = results["base_word"]
        
        for layer_idx in results["layers"]:
            if layer_idx in results["similarities"]:
                sim_data = results["similarities"][layer_idx]
                to_base = sim_data["to_base"]
                
                if len(to_base) > 1:
                    print(f"  Layer {layer_idx} - Word: {base_word}")
                    print(f"    Context progression: {to_base[0]:.3f} → {to_base[-1]:.3f}")
                    print(f"    Max context similarity: {max(to_base):.3f}")
                    print(f"    Min context similarity: {min(to_base):.3f}")
                    
                    # Identify largest drops in similarity
                    if len(sim_data["adjacent"]) > 0:
                        min_adjacent = min(sim_data["adjacent"])
                        min_idx = sim_data["adjacent"].index(min_adjacent)
                        print(f"    Largest context change: Level {min_idx} → {min_idx+1} (sim: {min_adjacent:.3f})")


def main():
    """Main analysis function."""
    print("Starting word embedding variation and contextual development analysis...")
    print(f"USE_SYSTEM_PROMPT_FOR_MANIFOLD: {USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
    if USE_SYSTEM_PROMPT_FOR_MANIFOLD:
        print(f"System prompt: '{SYSTEM_PROMPT}'")
    
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model_name_str = MODEL_NAME.split('/')[-1]
    
    # Define layers to analyze
    TARGET_LAYERS = [0, 15, 31]  # Early, middle, late layers
    
    # Results storage
    single_word_results = {}
    contextual_results = {}
    
    # Analyze each concept
    for concept, variations in WORD_VARIATIONS.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING CONCEPT: {concept.upper()}")
        print(f"{'='*80}")
        
        # Single-word variation analysis
        single_word_results[concept] = analyze_single_word_variations(
            model, tokenizer, concept, variations, TARGET_LAYERS
        )
        
        # Contextual development analysis (use the base word)
        base_word = variations[0]  # Use first variation as base word
        contextual_results[concept] = analyze_contextual_development(
            model, tokenizer, concept, base_word, TARGET_LAYERS
        )
        
        # Generate plots for this concept
        plot_word_variation_similarities(single_word_results[concept], f"single_word_{concept}")
        plot_contextual_development(contextual_results[concept], f"contextual_{concept}")
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print comprehensive summary
    print_analysis_summary(single_word_results, contextual_results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Generated plots:")
    for concept, variations in WORD_VARIATIONS.items():
        print(f"  - single_word_{concept}_{concept}_similarities.png")
        print(f"  - contextual_{concept}_{variations[0]}_progression.png")
        print(f"  - contextual_{concept}_{variations[0]}_context_matrix.png")


if __name__ == "__main__":
    main()
