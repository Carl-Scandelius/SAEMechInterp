#!/usr/bin/env python3
"""
Representation Lensing Experiment with target words {animals, furniture, food}.

Experiment: Extract embeddings from tokens BEFORE target words in Llama 3.1-8B (non-instruct).
Pass embeddings through language modeling head to get next-token predictions.
Perform local PCA (centered by target word mean) and perturb along PC1 with k multipliers.
Analyze and visualize results including cosine similarity matrices.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
from collections import defaultdict

from transformers import logging
logging.set_verbosity(40)

# Target words for the experiment  
TARGET_WORDS = {"animals", "furniture", "food"}
PERTURBATION_MULTIPLIERS = [0.5, 1.0, 2.0]


def load_sentences_with_target_words(target_words: Set[str], num_sentences: int = None) -> Dict[str, List[str]]:
    """
    Load sentences containing target words from manifold_sentences_hard_exactword_1000.json.
    
    Args:
        target_words: Set of target words to find
        num_sentences: Number of sentences to load per target word (if None, loads all available)
    
    Returns:
        Dict mapping target word to list of sentences containing it
    """
    sentences_by_target = {word: [] for word in target_words}
    
    # Load from manifold_sentences
    try:
        with open('../tools/manifold_sentences_hard_exactword_1000.json', 'r', encoding='utf-8') as f:
            manifold_data = json.load(f)
        
        for target_word in target_words:
            if target_word in manifold_data:
                if num_sentences is None:
                    # Load all available sentences
                    available_sentences = manifold_data[target_word]
                else:
                    # Load specified number of sentences
                    available_sentences = manifold_data[target_word][:num_sentences]
                    
                sentences_by_target[target_word].extend(available_sentences)
                print(f"   Loaded {len(available_sentences)} sentences with '{target_word}' from manifold_sentences.json")
            else:
                print(f"   Warning: '{target_word}' not found in manifold_sentences.json")
                
    except FileNotFoundError:
        raise FileNotFoundError("manifold_sentences_hard_exactword_1000.json not found in ../tools/")
    
    return sentences_by_target



def find_token_before_target_word(tokenizer: AutoTokenizer, sentence: str, target_word: str) -> Tuple[Optional[int], List[int]]:
    """
    Find the position of the token immediately before the target word's first token.
    
    Args:
        tokenizer: HuggingFace tokenizer
        sentence: Input sentence
        target_word: The target word to find
    
    Returns:
        Tuple of (position_before_target_or_None, all_token_ids)
    """
    # Tokenize the sentence
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Find the target word in the tokenized sequence
    target_word_lower = target_word.lower()
    
    for i, token in enumerate(tokens):
        # Clean the token (remove subword indicators like Ġ, ##, etc.)
        clean_token = token.replace('Ġ', '').replace('##', '').lower()
        
        # Check if this token starts or contains the target word
        if target_word_lower in clean_token or clean_token.startswith(target_word_lower):
            # Return the position before this token (if it exists)
            if i > 0:
                return i - 1, token_ids
            else:
                # Target word is the first token, no previous token
                return None, token_ids
    
    # Target word not found clearly, try a more flexible approach
    sentence_lower = sentence.lower()
    if target_word_lower in sentence_lower:
        # Find approximate position in the sentence
        word_start_pos = sentence_lower.find(target_word_lower)
        
        # Try to map this back to token positions
        current_pos = 0
        for i, token in enumerate(tokens):
            token_text = token.replace('Ġ', ' ').replace('##', '')
            if i > 0 and current_pos <= word_start_pos < current_pos + len(token_text):
                return max(0, i - 1), token_ids
            current_pos += len(token_text)
    
    return None, token_ids


class RepresentationLensingExtractor:
    """Extract embeddings from tokens before target words."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_pre_target_embeddings(self, sentences_by_target: Dict[str, List[str]], 
                                    layer_indices: List[int]) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Extract embeddings from tokens immediately before target words.
        
        Args:
            sentences_by_target: Dict mapping target word to sentences
            layer_indices: List of layer indices to extract from
        
        Returns:
            Dict[target_word][layer_idx] = tensor of embeddings
        """
        results = {target: {layer_idx: [] for layer_idx in layer_indices} 
                  for target in sentences_by_target.keys()}
        
        print("Extracting pre-target embeddings...")
        
        for target_word, sentences in sentences_by_target.items():
            print(f"\n  Processing target word: '{target_word}'")
            valid_extractions = 0
            
            for sentence_idx, sentence in enumerate(sentences):
                # Find position before target word
                pos_before_target, token_ids = find_token_before_target_word(
                    self.tokenizer, sentence, target_word
                )
                
                if pos_before_target is None:
                    print(f"    Skipping sentence {sentence_idx}: No token before '{target_word}'")
                    continue
                
                # Extract embeddings for each layer
                for layer_idx in layer_indices:
                    embedding = self._extract_embedding_at_position(
                        sentence, pos_before_target, layer_idx
                    )
                    if embedding is not None:
                        results[target_word][layer_idx].append(embedding)
                
                valid_extractions += 1
                
                if sentence_idx == 0:  # Debug first sentence
                    tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                    print(f"    Example: '{sentence[:50]}...'")
                    print(f"    Token at pos {pos_before_target}: '{tokens[pos_before_target]}'")
                    print(f"    Next token (target area): '{tokens[pos_before_target + 1] if pos_before_target + 1 < len(tokens) else 'END'}'")
            
            print(f"    Successfully extracted from {valid_extractions}/{len(sentences)} sentences")
            
                    # Convert lists to tensors
        for layer_idx in layer_indices:
            if results[target_word][layer_idx]:
                stacked_embeddings = torch.stack(results[target_word][layer_idx])
                results[target_word][layer_idx] = stacked_embeddings
                print(f"    Layer {layer_idx}: {stacked_embeddings.shape} dtype: {stacked_embeddings.dtype}")
            else:
                print(f"    Layer {layer_idx}: No valid embeddings extracted!")
        
        return results
    
    def _extract_embedding_at_position(self, sentence: str, position: int, layer_idx: int) -> Optional[torch.Tensor]:
        """Extract embedding at specific position and layer."""
        activations = []
        
        def hook_fn(module, input, output):
            # Get hidden states from the transformer layer
            if isinstance(output, tuple):
                hidden_states = output[0]  # (batch_size, seq_len, hidden_size)
            else:
                hidden_states = output
            
            # Extract embedding at the specified position
            if position < hidden_states.shape[1]:
                # Preserve the original dtype when moving to CPU
                embedding = hidden_states[0, position, :].detach().cpu()
                activations.append(embedding)
        
        # Register hook on the transformer layer
        layer_module = self.model.get_submodule(f"model.layers.{layer_idx}")
        hook = layer_module.register_forward_hook(hook_fn)
        
        try:
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            if activations:
                return activations[0]
            else:
                return None
                
        finally:
            hook.remove()


class LanguageModelingAnalyzer:
    """Analyze next-token predictions using language modeling head."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.lm_head = model.lm_head
        
    def get_top_predictions(self, embeddings: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Pass embeddings through language modeling head and get top predictions.
        
        Args:
            embeddings: Tensor of embeddings (batch_size, hidden_size)
            top_k: Number of top predictions to return
        
        Returns:
            List of (token_string, probability) tuples
        """
        # Ensure embeddings match the model's dtype and device
        embeddings = embeddings.to(self.device, dtype=self.lm_head.weight.dtype)
        
        with torch.no_grad():
            # Pass through language modeling head
            logits = self.lm_head(embeddings)  # (batch_size, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            
            # Get top-k predictions across all examples
            mean_probs = probs.mean(dim=0)  # (vocab_size,)
            top_k_values, top_k_indices = torch.topk(mean_probs, top_k)
            
            # Convert to tokens and probabilities
            predictions = []
            for value, idx in zip(top_k_values, top_k_indices):
                token = self.tokenizer.decode([idx.item()])
                predictions.append((token, value.item()))
        
        return predictions
    
    def analyze_predictions_by_target(self, embeddings_by_target: Dict[str, Dict[int, torch.Tensor]], 
                                    layers: List[int]) -> Dict[str, Dict[int, List[Tuple[str, float]]]]:
        """Analyze predictions for each target word and layer."""
        results = {}
        
        print("\nAnalyzing next-token predictions...")
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\n  Target word: '{target_word}'")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                if len(embeddings) > 0:
                    predictions = self.get_top_predictions(embeddings, top_k=3)
                    results[target_word][layer_idx] = predictions
                    
                    print(f"    Layer {layer_idx} top predictions:")
                    for i, (token, prob) in enumerate(predictions):
                        print(f"      {i+1}. '{token}' (p={prob:.4f})")
                else:
                    results[target_word][layer_idx] = []
                    print(f"    Layer {layer_idx}: No embeddings to analyze")
        
        return results


class LocalPCAAnalyzer:
    """Perform local PCA analysis centered on target word embeddings."""
    
    def compute_local_pca(self, embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute local PCA by centering embeddings on their mean.
        
        Args:
            embeddings: Tensor of embeddings (num_samples, hidden_size)
        
        Returns:
            Tuple of (components, eigenvalues, explained_variance_ratios)
        """
        if len(embeddings) < 2:
            print("Warning: Not enough embeddings for PCA")
            return np.array([]), np.array([]), np.array([])
        
        # Convert to numpy and center the data
        embeddings_np = embeddings.cpu().numpy()
        mean_embedding = np.mean(embeddings_np, axis=0)
        centered_embeddings = embeddings_np - mean_embedding
        
        # Compute PCA
        # Use min(n_samples-1, n_features) to avoid issues with small datasets
        n_components = min(centered_embeddings.shape[0] - 1, centered_embeddings.shape[1], 10)
        
        if n_components <= 0:
            print("Warning: Cannot compute PCA with current data")
            return np.array([]), np.array([]), np.array([])
        
        pca = PCA(n_components=n_components)
        pca.fit(centered_embeddings)
        
        return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_
    
    def analyze_all_targets(self, embeddings_by_target: Dict[str, Dict[int, torch.Tensor]], 
                          layers: List[int]) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
        """Compute local PCA for all target words and layers."""
        results = {}
        
        print("\nPerforming local PCA analysis...")
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\n  Target word: '{target_word}'")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                if len(embeddings) > 1:
                    components, eigenvalues, explained_variance = self.compute_local_pca(embeddings)
                    
                    results[target_word][layer_idx] = {
                        'components': components,
                        'eigenvalues': eigenvalues,
                        'explained_variance': explained_variance
                    }
                    
                    if len(explained_variance) > 0:
                        print(f"    Layer {layer_idx}: Top PC explains {explained_variance[0]:.4f} of variance")
                    else:
                        print(f"    Layer {layer_idx}: PCA computation failed")
                else:
                    results[target_word][layer_idx] = {
                        'components': np.array([]),
                        'eigenvalues': np.array([]),
                        'explained_variance': np.array([])
                    }
                    print(f"    Layer {layer_idx}: Insufficient data for PCA")
        
        return results


class PerturbationAnalyzer:
    """Analyze effects of perturbing embeddings along principal components."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, lm_analyzer: LanguageModelingAnalyzer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.lm_analyzer = lm_analyzer
    
    def perturb_and_analyze(self, embeddings_by_target: Dict[str, Dict[int, torch.Tensor]],
                          pca_results: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                          layers: List[int],
                          k_values: List[float] = [0.5, 1.0, 2.0]) -> Dict:
        """
        Perturb embeddings along PC1 and analyze resulting predictions.
        
        Args:
            embeddings_by_target: Original embeddings
            pca_results: PCA analysis results
            layers: Layer indices
            k_values: Multipliers for perturbation
        
        Returns:
            Dict with perturbation analysis results
        """
        results = {}
        
        print("\nPerforming perturbation analysis...")
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\n  Target word: '{target_word}'")
            
            for layer_idx in layers:
                results[target_word][layer_idx] = {}
                
                embeddings = embeddings_by_target[target_word][layer_idx]
                pca_data = pca_results[target_word][layer_idx]
                
                if len(embeddings) == 0 or len(pca_data['components']) == 0:
                    print(f"    Layer {layer_idx}: No data for perturbation")
                    continue
                
                # Get the first principal component
                pc1 = pca_data['components'][0]  # Shape: (hidden_size,)
                eigenvalue1 = pca_data['eigenvalues'][0] if len(pca_data['eigenvalues']) > 0 else 1.0
                
                print(f"    Layer {layer_idx}: Perturbing along PC1 (eigenvalue: {eigenvalue1:.4f})")
                
                # Center the embeddings (as in PCA)
                embeddings_np = embeddings.cpu().numpy()
                mean_embedding = np.mean(embeddings_np, axis=0)
                centered_embeddings = embeddings_np - mean_embedding
                
                # Perturb along PC1 with different k values
                for k in k_values:
                    perturbation = k * eigenvalue1 * pc1
                    perturbed_embeddings = centered_embeddings + perturbation
                    
                    # Add back the mean to get absolute embeddings
                    final_embeddings = perturbed_embeddings + mean_embedding
                    
                    # Analyze predictions - preserve the original embeddings' dtype
                    perturbed_tensor = torch.from_numpy(final_embeddings).to(dtype=embeddings.dtype)
                    predictions = self.lm_analyzer.get_top_predictions(perturbed_tensor, top_k=3)
                    
                    results[target_word][layer_idx][f'k_{k}'] = predictions
                    
                    print(f"      k={k}: {predictions[0][0]} (p={predictions[0][1]:.4f})")
        
        return results


def compute_pc_cosine_similarity_matrix(pca_results: Dict[str, Dict[int, Dict[str, np.ndarray]]],
                                      layers: List[int]) -> Tuple[np.ndarray, List[str]]:
    """Compute cosine similarity matrix between all PC1s across target words and layers."""
    
    all_pc1s = []
    labels = []
    
    for target_word in sorted(pca_results.keys()):
        for layer_idx in layers:
            pca_data = pca_results[target_word][layer_idx]
            if len(pca_data['components']) > 0:
                pc1 = pca_data['components'][0]
                all_pc1s.append(pc1)
                labels.append(f"{target_word}_L{layer_idx}")
    
    if len(all_pc1s) == 0:
        return np.array([]), []
    
    # Compute cosine similarity matrix
    pc_matrix = np.stack(all_pc1s)
    
    # Normalize vectors
    norms = np.linalg.norm(pc_matrix, axis=1, keepdims=True)
    normalized = pc_matrix / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix, labels


def visualize_results(baseline_predictions: Dict, perturbation_results: Dict, 
                     similarity_matrix: np.ndarray, similarity_labels: List[str],
                     pca_results: Dict, output_dir: str = "rep_lensing_results"):
    """Create visualizations for the experiment results."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nCreating visualizations in {output_dir}/...")
    
    # 1. PC Cosine Similarity Matrix
    if len(similarity_matrix) > 0:
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                   xticklabels=similarity_labels, 
                   yticklabels=similarity_labels,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.3f',
                   square=True)
        plt.title('Cosine Similarity Matrix: PC1s Across Target Words and Layers')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pc1_cosine_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved PC cosine similarity matrix")
    
    # 2. Explained Variance Summary
    fig, axes = plt.subplots(1, len(TARGET_WORDS), figsize=(15, 5))
    if len(TARGET_WORDS) == 1:
        axes = [axes]
    
    for idx, target_word in enumerate(sorted(TARGET_WORDS)):
        layers = []
        variances = []
        
        for layer_key in sorted(pca_results[target_word].keys()):
            pca_data = pca_results[target_word][layer_key]
            if len(pca_data['explained_variance']) > 0:
                layers.append(layer_key)
                variances.append(pca_data['explained_variance'][0])
        
        if layers:
            axes[idx].bar(range(len(layers)), variances)
            axes[idx].set_title(f"PC1 Explained Variance: '{target_word}'")
            axes[idx].set_xlabel('Layer')
            axes[idx].set_ylabel('Explained Variance Ratio')
            axes[idx].set_xticks(range(len(layers)))
            axes[idx].set_xticklabels([f'L{l}' for l in layers])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/explained_variance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved explained variance summary")


def run_representation_lensing_experiment(num_sentences: int = None, 
                                        layers: List[int] = [0, 15, 31]):
    """Run the complete representation lensing experiment."""
    
    output_dir = "rep_lensing_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=== REPRESENTATION LENSING EXPERIMENT ===")
    print("Model: Llama 3.1-8B (non-instruct)")
    print(f"Target words: {TARGET_WORDS}")
    print(f"Layers: {layers}")
    if num_sentences is None:
        print("Loading all available sentences from manifold_sentences_hard_exactword_1000.json")
    else:
        print(f"Sentences per target word: {num_sentences}")
    print(f"Perturbation multipliers: {PERTURBATION_MULTIPLIERS}")
    
    # 1. Load model and tokenizer
    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",  # Non-instruct version
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load sentences
    print("\n2. Loading sentences with target words...")
    sentences_by_target = load_sentences_with_target_words(TARGET_WORDS, num_sentences)
    
    # 3. Extract embeddings
    print("\n3. Extracting pre-target embeddings...")
    extractor = RepresentationLensingExtractor(model, tokenizer)
    embeddings_by_target = extractor.extract_pre_target_embeddings(sentences_by_target, layers)
    
    # 4. Analyze baseline predictions
    print("\n4. Analyzing baseline next-token predictions...")
    lm_analyzer = LanguageModelingAnalyzer(model, tokenizer)
    baseline_predictions = lm_analyzer.analyze_predictions_by_target(embeddings_by_target, layers)
    
    # 5. Perform local PCA
    print("\n5. Performing local PCA analysis...")
    pca_analyzer = LocalPCAAnalyzer()
    pca_results = pca_analyzer.analyze_all_targets(embeddings_by_target, layers)
    
    # 6. Perturbation analysis
    print("\n6. Running perturbation experiments...")
    perturbation_analyzer = PerturbationAnalyzer(model, tokenizer, lm_analyzer)
    perturbation_results = perturbation_analyzer.perturb_and_analyze(
        embeddings_by_target, pca_results, layers, PERTURBATION_MULTIPLIERS
    )
    
    # 7. Compute similarity matrix
    print("\n7. Computing PC cosine similarity matrix...")
    similarity_matrix, similarity_labels = compute_pc_cosine_similarity_matrix(pca_results, layers)
    
    # 8. Save results
    print("\n8. Saving results...")
    results = {
        'experiment_config': {
            'target_words': list(TARGET_WORDS),
            'layers': layers,
            'num_sentences': num_sentences if num_sentences is not None else 'all_available',
            'perturbation_multipliers': PERTURBATION_MULTIPLIERS
        },
        'baseline_predictions': baseline_predictions,
        'perturbation_results': perturbation_results,
        'pca_explained_variance': {
            target: {
                str(layer): {
                    'top_pc_variance': float(pca_results[target][layer]['explained_variance'][0]) 
                    if len(pca_results[target][layer]['explained_variance']) > 0 else 0.0
                }
                for layer in layers
            }
            for target in TARGET_WORDS
        },
        'pc_similarity_matrix': similarity_matrix.tolist() if len(similarity_matrix) > 0 else [],
        'pc_similarity_labels': similarity_labels
    }
    
    with open(f'{output_dir}/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Saved results to {output_dir}/experiment_results.json")
    
    # 9. Create visualizations
    print("\n9. Creating visualizations...")
    visualize_results(baseline_predictions, perturbation_results, 
                     similarity_matrix, similarity_labels, pca_results, output_dir)
    
    print("\n=== EXPERIMENT COMPLETE ===")
    print(f"\nResults Summary:")
    print(f"- Analyzed {len(TARGET_WORDS)} target words across {len(layers)} layers")
    print(f"- Results saved to {output_dir}/")
    
    # Print key findings
    print(f"\nKey Findings:")
    for target_word in TARGET_WORDS:
        print(f"\n'{target_word.upper()}':")
        for layer_idx in layers:
            if len(pca_results[target_word][layer_idx]['explained_variance']) > 0:
                top_variance = pca_results[target_word][layer_idx]['explained_variance'][0]
                print(f"  Layer {layer_idx}: PC1 explains {top_variance:.3f} of variance")
                
                if layer_idx in baseline_predictions[target_word]:
                    top_pred = baseline_predictions[target_word][layer_idx][0]
                    print(f"    Baseline top prediction: '{top_pred[0]}' (p={top_pred[1]:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run representation lensing experiment.")
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=100,
        help="Number of sentences to load per target word"
    )
    parser.add_argument(
        "--layers",
        nargs='+',
        type=int,
        default=[0, 5, 15, 25, 31],
        help="Layer indices to analyze"
    )
    
    args = parser.parse_args()
    
    run_representation_lensing_experiment(
        num_sentences=args.num_sentences,
        layers=args.layers
    ) 