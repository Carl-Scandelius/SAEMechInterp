#!/usr/bin/env python3
"""
Prediction steering experiment with dog/bird categories.

Experiment: Extract final token embeddings from custom dataset on Llama 3.1-8B-Instruct.
Find PCs of these embeddings for each layer. Keep top 3 by explained variance.
Steer model output by perturbing final token in direction of top PCs.
Generate cosine similarity matrix of all retained PCs across layers.
Add centroid-based perturbation experiments.
"""

import torch
import numpy as np
import requests
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from transformers import logging
logging.set_verbosity(40)

def load_prediction_dataset() -> Tuple[List[str], List[str]]:
    """
    Load the prediction dataset from local JSON file.

    Returns:
        A tuple of (formatted_sentences, labels).
    """
    with open('prediction_sent.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    labels = []
    
    for category, sentence_list in data.items():
        for sentence in sentence_list:
            # Add the specified formatting to each sentence
            formatted_sentence = f'Fill in the word that has been replaced by the asterisk (*) symbol in the following sentence, responding only with the replaced word in all lowercase, singular form: "{sentence}". The missing word is:'
            sentences.append(formatted_sentence)
            labels.append(category)
    
    return sentences, labels


class MinimalAttentionExtractor:
    """Extract final token embeddings from a specified model component."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, use_residual_stream: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.use_residual_stream = use_residual_stream
        
    def extract_final_token_embeddings(self, prompts: List[str], layer_indices: List[int]) -> Dict[int, torch.Tensor]:
        """Extract final content token embeddings from attention or residual stream."""
        results = {layer_idx: [] for layer_idx in layer_indices}
        
        for layer_idx in layer_indices:
            activations = []
            
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attention_output = output[0]
                else:
                    attention_output = output
                
                # Extract the final content token (avoiding special end tokens)
                # For now, use -1 but we'll add debugging to see what tokens we're getting
                final_content_token = attention_output[:, -1, :].detach().cpu()
                activations.append(final_content_token)
            
            # Register hook
            if self.use_residual_stream:
                module = self.model.get_submodule(f"model.layers.{layer_idx}")
            else:
                module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
            hook = module.register_forward_hook(hook_fn)
            
            try:
                for prompt in prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    # Debug: Print tokenization to understand the sequence
                    if layer_idx == 0 and len(activations) == 0:  # Only print once
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                        print(f"   Sample tokenization: ...{tokens[-5:]} (extracting position -1: '{tokens[-1]}')")
                        print(f"   Full prompt: {prompt[:100]}...")
                        print(f"   Final tokens: {tokens[-10:]}")
                    
                    with torch.no_grad():
                        self.model(**inputs)
            finally:
                hook.remove()
            
            results[layer_idx] = torch.cat(activations, dim=0)
        
        return results


class MinimalPCAAnalyzer:
    """Compute PCA on embedding data."""
    
    def compute_pca_top3(self, embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute top 3 principal components."""
        pca = PCA(n_components=3)
        embeddings_np = embeddings.cpu().numpy()
        pca.fit(embeddings_np)
        
        return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_


class MinimalSteerer:
    """Steer model by perturbing embeddings at a specified component."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, use_residual_stream: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.use_residual_stream = use_residual_stream
        
    def steer_generation(self, prompt: str, layer_idx: int, pc_direction: torch.Tensor, 
                        eigenvalue: float, multiple: float = 1.0, max_new_tokens: int = 50,
                        perturb_all_tokens: bool = False) -> str:
        """Generate text with final token steering."""
        perturbation_applied = False
        
        def steering_hook(module, input, output):
            nonlocal perturbation_applied
            if perturbation_applied:
                return output
                
            attention_output = output[0] if isinstance(output, tuple) else output
            
            # Apply perturbation to final token or all tokens
            perturbation = multiple * eigenvalue * pc_direction.to(self.device, dtype=attention_output.dtype)
            if perturb_all_tokens:
                attention_output[:, :, :] += perturbation
            else:
                attention_output[:, -1, :] += perturbation
            
            perturbation_applied = True
            
            if isinstance(output, tuple):
                return (attention_output,) + output[1:]
            return attention_output
        
        # Register hook
        if self.use_residual_stream:
            module = self.model.get_submodule(f"model.layers.{layer_idx}")
        else:
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
        hook = module.register_forward_hook(steering_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        finally:
            hook.remove()
    
    def steer_generation_with_centroid(self, prompt: str, layer_idx: int, centroid: torch.Tensor, 
                                      replacement_mode: bool = False, progress: float = 1.0,
                                      max_new_tokens: int = 50, perturb_all_tokens: bool = False) -> str:
        """Generate text with centroid-based steering."""
        perturbation_applied = False
        
        def centroid_hook(module, input, output):
            nonlocal perturbation_applied
            if perturbation_applied:
                return output
                
            attention_output = output[0] if isinstance(output, tuple) else output
            
            if replacement_mode:
                # Replace with centroid
                if perturb_all_tokens:
                    attention_output[:, :, :] = centroid.to(attention_output.device, dtype=attention_output.dtype)
                else:
                    attention_output[:, -1, :] = centroid.to(attention_output.device, dtype=attention_output.dtype)
            else:
                # Move toward centroid by progress amount
                current_embedding = attention_output[:, -1, :] if not perturb_all_tokens else attention_output
                direction = centroid.to(attention_output.device, dtype=attention_output.dtype) - current_embedding
                perturbation = direction * progress
                
                if perturb_all_tokens:
                    attention_output[:, :, :] += perturbation
                else:
                    attention_output[:, -1, :] += perturbation
            
            perturbation_applied = True
            
            if isinstance(output, tuple):
                return (attention_output,) + output[1:]
            return attention_output
        
        # Register hook
        if self.use_residual_stream:
            module = self.model.get_submodule(f"model.layers.{layer_idx}")
        else:
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
        hook = module.register_forward_hook(centroid_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        finally:
            hook.remove()

    def steer_generation_with_opposite_token(self, prompt: str, layer_idx: int, opposite_embedding: torch.Tensor, 
                                            max_new_tokens: int = 50, perturb_all_tokens: bool = False) -> str:
        """Generate text by replacing the final token with an embedding from the opposite category."""
        perturbation_applied = False
        
        def replacement_hook(module, input, output):
            nonlocal perturbation_applied
            if perturbation_applied:
                return output
                
            attention_output = output[0] if isinstance(output, tuple) else output
            
            # Replace final token (or all tokens) with the opposite category embedding
            if perturb_all_tokens:
                attention_output[:, :, :] = opposite_embedding.to(attention_output.device, dtype=attention_output.dtype)
            else:
                attention_output[:, -1, :] = opposite_embedding.to(attention_output.device, dtype=attention_output.dtype)
            
            perturbation_applied = True
            
            if isinstance(output, tuple):
                return (attention_output,) + output[1:]
            return attention_output
        
        # Register hook
        if self.use_residual_stream:
            module = self.model.get_submodule(f"model.layers.{layer_idx}")
        else:
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
        hook = module.register_forward_hook(replacement_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        finally:
            hook.remove()


def compute_pc_similarity_matrix(all_pcs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Compute cosine similarity matrix between all PCs across layers."""
    all_vectors = []
    labels = []
    
    for layer_idx in sorted(all_pcs.keys()):
        for pc_idx in range(len(all_pcs[layer_idx])):
            all_vectors.append(all_pcs[layer_idx][pc_idx])
            labels.append(f"L{layer_idx}_PC{pc_idx}")
    
    all_vectors = np.array(all_vectors)
    normalized = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix, labels


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    title: str,
    output_dir: str = "prediction_steering_results"
):
    """Visualize a cosine similarity matrix as a heatmap and save it."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    # Sanitize filename
    filename_title = ''.join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
    filename = f"{output_dir}/{filename_title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    print(f"   Saved similarity matrix to '{filename}'")
    plt.close()


def plot_embedding_space_2d(
    embeddings: torch.Tensor,
    layer_idx: int,
    cluster_labels: Optional[np.ndarray] = None,
    sentence_labels: Optional[List[str]] = None,
    output_dir: str = "prediction_steering_results",
    title_prefix: str = ""
):
    """
    Visualize the embedding space using PCA and save the plot.
    """
    print(f"   Visualizing embedding space for layer {layer_idx}...")
    
    # 1. Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings.numpy())
    
    # 2. Create plot
    plt.figure(figsize=(12, 10))
    
    plot_suffix = ""
    title_suffix = ""
    
    if sentence_labels is not None:
        # Plot with sentence type colors
        df = pd.DataFrame({'PC1': embeddings_2d[:, 0], 'PC2': embeddings_2d[:, 1], 'label': sentence_labels})
        sns.scatterplot(
            data=df,
            x='PC1',
            y='PC2',
            hue='label',
            palette="viridis",
            legend="full",
            alpha=0.7
        )
        title_suffix = " by Category"
        plot_suffix = "_by_category"
    elif cluster_labels is not None:
        # Plot with cluster colors
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=cluster_labels,
            palette="viridis",
            legend="full",
            alpha=0.7
        )
        title_suffix = " with K-Means Clusters"
        plot_suffix = "_clusters"
    else:
        # Plot without clusters
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            alpha=0.7
        )
        
    plt.title(f'{title_prefix}Embedding Space - Layer {layer_idx}{title_suffix}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = f"{output_dir}/{title_prefix.lower().replace(' ', '_')}embedding_space_layer_{layer_idx}{plot_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved embedding space visualization to '{filename}'")
    plt.close()


def plot_embedding_space_3d(
    embeddings: torch.Tensor,
    layer_idx: int,
    sentence_labels: List[str],
    output_dir: str = "prediction_steering_results",
    title_prefix: str = ""
):
    """Visualize the embedding space in 3D using PCA."""
    print(f"   Visualizing 3D embedding space for layer {layer_idx}...")
    
    # 1. Reduce to 3D using PCA on the full set of embeddings
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings.numpy())
    
    # 2. Prepare data for plotting
    df = pd.DataFrame({
        'PC1': embeddings_3d[:, 0],
        'PC2': embeddings_3d[:, 1],
        'PC3': embeddings_3d[:, 2],
        'Category': sentence_labels
    })

    # 3. Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    markers = {'dog': 'o', 'bird': '^'}
    colors = {'dog': 'blue', 'bird': 'red'}

    for category, group in df.groupby('Category'):
        ax.scatter(
            group['PC1'], group['PC2'], group['PC3'],
            label=category,
            marker=markers.get(category, 'd'),
            c=[colors.get(category, 'black')],
            s=50,
            alpha=0.7
        )

    ax.set_title(f'{title_prefix}3D Embedding Space - Layer {layer_idx}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = f"{output_dir}/{title_prefix.lower().replace(' ', '_')}embedding_space_3d_layer_{layer_idx}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved 3D embedding space visualization to '{filename}'")
    plt.close()


def plot_embedding_space_tsne(
    embeddings: torch.Tensor,
    layer_idx: int,
    sentence_labels: List[str],
    output_dir: str = "prediction_steering_results",
    title_prefix: str = ""
):
    """Visualize the embedding space in 2D using t-SNE."""
    print(f"   Visualizing t-SNE embedding space for layer {layer_idx}...")

    # 1. Reduce to 2D using t-SNE
    # Use a smaller subset for t-SNE if the dataset is large, as it can be slow
    sample_size = min(1000, embeddings.shape[0])
    indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    sampled_embeddings = embeddings.numpy()[indices]
    sampled_labels = [sentence_labels[i] for i in indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size - 1))
    embeddings_2d = tsne.fit_transform(sampled_embeddings)

    # 2. Prepare data for plotting
    df = pd.DataFrame({
        'Component 1': embeddings_2d[:, 0],
        'Component 2': embeddings_2d[:, 1],
        'Category': sampled_labels
    })

    # 3. Create 2D plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x='Component 1',
        y='Component 2',
        hue='Category',
        style='Category',
        palette="viridis",
        s=100,
        alpha=0.8
    )

    plt.title(f'{title_prefix}t-SNE Embedding Space - Layer {layer_idx}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = f"{output_dir}/{title_prefix.lower().replace(' ', '_')}embedding_space_tsne_layer_{layer_idx}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved t-SNE embedding space visualization to '{filename}'")
    plt.close()


def run_prediction_experiment(perturb_all_tokens: bool = False, use_residual_stream: bool = False):
    """Run the complete prediction steering experiment."""
    
    output_dir = "prediction_steering_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=== PREDICTION STEERING EXPERIMENT ===")
    print("Model: Llama 3.1-8B-Instruct")
    print("Layers: [0, 15, 31] (first, middle, last)")
    print("Dataset: Custom dog/bird prediction dataset")
    if use_residual_stream:
        print("Intervention point: Residual stream (layer output)")
    else:
        print("Intervention point: Attention block output (pre-residual)")
    
    # 1. Load model and tokenizer
    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load dataset
    print("2. Loading prediction dataset...")
    sentences, labels = load_prediction_dataset()
    print(f"   Loaded {len(sentences)} sentences")
    
    # 3. Extract embeddings
    print("3. Extracting final token embeddings...")
    layers = [0, 15, 31]
    extractor = MinimalAttentionExtractor(model, tokenizer, use_residual_stream=use_residual_stream)
    embeddings = extractor.extract_final_token_embeddings(sentences, layers)
    
    for layer_idx, emb in embeddings.items():
        print(f"   Layer {layer_idx}: {emb.shape}")
    
    # 4. Compute PCA (top 3 components per layer)
    print("4. Computing PCA (top 3 components per layer)...")
    analyzer = MinimalPCAAnalyzer()
    
    all_pcs = {}
    all_eigenvalues = {}
    all_explained_variance = {}
    
    for layer_idx in layers:
        pcs, eigenvals, explained_var = analyzer.compute_pca_top3(embeddings[layer_idx])
        all_pcs[layer_idx] = pcs
        all_eigenvalues[layer_idx] = eigenvals
        all_explained_variance[layer_idx] = explained_var
        
        print(f"   Layer {layer_idx}: Explained variance = {explained_var}")
    
    # Visualize embedding spaces
    print("\nVisualizing embedding spaces...")
    for layer_idx in layers:
        plot_embedding_space_2d(embeddings[layer_idx], layer_idx, sentence_labels=labels, output_dir=output_dir)
        plot_embedding_space_3d(embeddings[layer_idx], layer_idx, labels, output_dir)
        plot_embedding_space_tsne(embeddings[layer_idx], layer_idx, labels, output_dir)

    # 5. Steering experiments
    print("5. Running steering experiments...")
    steerer = MinimalSteerer(model, tokenizer, use_residual_stream=use_residual_stream)
    
    test_prompt = "Fill in the word that has been replaced by the asterisk (*) symbol in the following sentence, responding only with the replaced word in all lowercase, singular form. \"A * is a four-legged member of the Canidae family often kept as a household pet.\""
    multiples = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    
    print(f"\nTest prompt: '{test_prompt}'")
    
    # Baseline
    baseline = steerer.steer_generation(test_prompt, 0, torch.zeros(embeddings[0].shape[1]), 0, 0, perturb_all_tokens=perturb_all_tokens)
    print(f"\nBaseline: {baseline}")
    
    # Define clusters and centroids
    dog_mask = np.array(labels) == 'dog'
    bird_mask = np.array(labels) == 'bird'
    
    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")
        
        dog_embeddings = embeddings[layer_idx][dog_mask]
        bird_embeddings = embeddings[layer_idx][bird_mask]
        
        dog_centroid = dog_embeddings.mean(dim=0)
        bird_centroid = bird_embeddings.mean(dim=0)
        
        print(f"Dog cluster size: {dog_embeddings.shape[0]}")
        print(f"Bird cluster size: {bird_embeddings.shape[0]}")
        
        # Euclidean distance between centroids
        distance = torch.norm(dog_centroid - bird_centroid).item()
        print(f"Distance between centroids: {distance:.4f}")
        
        # Global PC steering
        print(f"\nGlobal PC steering (Layer {layer_idx}):")
        for pc_idx in [0, 1]:  # Top 2 PCs
            pc_direction = torch.from_numpy(all_pcs[layer_idx][pc_idx])
            eigenvalue = all_eigenvalues[layer_idx][pc_idx]
            
            print(f"\nPC {pc_idx} (eigenvalue: {eigenvalue:.4f}):")
            for multiple in multiples:
                steered = steerer.steer_generation(
                    test_prompt, layer_idx, pc_direction, eigenvalue, multiple,
                    perturb_all_tokens=perturb_all_tokens
                )
                print(f"  {multiple:+.1f}x: {steered}")
        
        # Local PC steering for each category
        for category, category_embeddings in [("dog", dog_embeddings), ("bird", bird_embeddings)]:
            if len(category_embeddings) >= 3:
                local_pcs, local_eigenvals, local_explained_var = analyzer.compute_pca_top3(category_embeddings)
                
                print(f"\nLocal {category} PC steering (Layer {layer_idx}):")
                pc_direction = torch.from_numpy(local_pcs[0])
                eigenvalue = local_eigenvals[0]
                
                print(f"PC0 (eigenvalue: {eigenvalue:.4f}):")
                for multiple in multiples:
                    steered = steerer.steer_generation(
                        test_prompt, layer_idx, pc_direction, eigenvalue, multiple,
                        perturb_all_tokens=perturb_all_tokens
                    )
                    print(f"  {multiple:+.1f}x: {steered}")
        
        # NEW: Centroid-based perturbations
        print(f"\nCentroid-based perturbations (Layer {layer_idx}):")
        
        # Move toward opposite centroid by 10% intervals
        print("\nMoving toward bird centroid (from dog perspective):")
        for progress in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            steered = steerer.steer_generation_with_centroid(
                test_prompt, layer_idx, bird_centroid, replacement_mode=False, 
                progress=progress, perturb_all_tokens=perturb_all_tokens
            )
            print(f"  {progress*100:3.0f}%: {steered}")
        
        print("\nMoving toward dog centroid (from bird perspective):")
        for progress in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            steered = steerer.steer_generation_with_centroid(
                test_prompt, layer_idx, dog_centroid, replacement_mode=False, 
                progress=progress, perturb_all_tokens=perturb_all_tokens
            )
            print(f"  {progress*100:3.0f}%: {steered}")
        
        # Replace with centroid
        print("\nReplacing with bird centroid:")
        steered = steerer.steer_generation_with_centroid(
            test_prompt, layer_idx, bird_centroid, replacement_mode=True,
            perturb_all_tokens=perturb_all_tokens
        )
        print(f"  Bird replacement: {steered}")
        
        print("\nReplacing with dog centroid:")
        steered = steerer.steer_generation_with_centroid(
            test_prompt, layer_idx, dog_centroid, replacement_mode=True,
            perturb_all_tokens=perturb_all_tokens
        )
        print(f"  Dog replacement: {steered}")
        
        # NEW: Opposite token replacement experiments
        print(f"\nOpposite token replacement experiments (Layer {layer_idx}):")
        
        # Get the final token embeddings from specific sentences of opposite categories
        if len(dog_embeddings) > 0 and len(bird_embeddings) > 0:
            # Select one specific sentence from each category for swapping
            # These represent the final token embeddings from actual sentences
            bird_sentence_idx = 0  # First bird sentence
            dog_sentence_idx = 0   # First dog sentence
            
            if bird_sentence_idx < len(bird_embeddings):
                bird_final_token = bird_embeddings[bird_sentence_idx]
                print(f"\nReplacing test_prompt final token with bird sentence {bird_sentence_idx} final token:")
                steered = steerer.steer_generation_with_opposite_token(
                    test_prompt, layer_idx, bird_final_token,
                    perturb_all_tokens=perturb_all_tokens
                )
                print(f"  Result: {steered}")
            
            if dog_sentence_idx < len(dog_embeddings):
                dog_final_token = dog_embeddings[dog_sentence_idx]
                print(f"\nReplacing test_prompt final token with dog sentence {dog_sentence_idx} final token:")
                steered = steerer.steer_generation_with_opposite_token(
                    test_prompt, layer_idx, dog_final_token,
                    perturb_all_tokens=perturb_all_tokens
                )
                print(f"  Result: {steered}")
                
            # Also try with a few more examples for robustness
            for additional_idx in [1, 2]:
                if additional_idx < len(bird_embeddings):
                    bird_final_token = bird_embeddings[additional_idx]
                    print(f"\nReplacing test_prompt final token with bird sentence {additional_idx} final token:")
                    steered = steerer.steer_generation_with_opposite_token(
                        test_prompt, layer_idx, bird_final_token,
                        perturb_all_tokens=perturb_all_tokens
                    )
                    print(f"  Result: {steered}")
                
                if additional_idx < len(dog_embeddings):
                    dog_final_token = dog_embeddings[additional_idx]
                    print(f"\nReplacing test_prompt final token with dog sentence {additional_idx} final token:")
                    steered = steerer.steer_generation_with_opposite_token(
                        test_prompt, layer_idx, dog_final_token,
                        perturb_all_tokens=perturb_all_tokens
                    )
                    print(f"  Result: {steered}")

    # 6. Compute similarity matrix
    print("\n6. Computing PC similarity matrix...")
    similarity_matrix, labels_list = compute_pc_similarity_matrix(all_pcs)
    
    # Plot similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=labels_list, yticklabels=labels_list)
    plt.title('PC Cosine Similarity Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pc_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Saved similarity matrix to '{output_dir}/pc_similarity_matrix.png'")
    
    # Save numerical results
    results = {
        'all_eigenvalues': {k: v.tolist() for k, v in all_eigenvalues.items()},
        'all_explained_variance': {k: v.tolist() for k, v in all_explained_variance.items()},
        'similarity_matrix': similarity_matrix.tolist(),
        'similarity_labels': labels_list
    }
    
    with open(f'{output_dir}/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved numerical results to '{output_dir}/experiment_results.json'")
    
    print("\n=== EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction steering experiment.")
    parser.add_argument(
        "--perturb-all-tokens",
        action="store_true",
        help="Apply steering perturbation to all tokens, not just the final token."
    )
    parser.add_argument(
        "--use-residual-stream",
        action="store_true",
        help="Extract embeddings and apply perturbations to the residual stream (output of a layer) instead of just the attention block output."
    )
    args = parser.parse_args()
    
    run_prediction_experiment(
        perturb_all_tokens=args.perturb_all_tokens,
        use_residual_stream=args.use_residual_stream
    ) 