#!/usr/bin/env python3
"""
Representation Lensing Experiment for Llama 3.1-8B (non-instruct)

This experiment:
1. Extracts embeddings from tokens immediately before target words
2. Analyzes next-token predictions using three projection methods:
   - lm_head: Standard language modeling head
   - embedding_transpose: Transpose of embedding matrix
   - lm_head_with_norm: LM head with final RMSNorm (complete pipeline)
3. Performs local PCA on embeddings and tests perturbations
4. Tests multi-centroid perturbations (each category perturbed by all three centroids)
5. Generates cosine similarity matrices and prediction analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import argparse
import os
from pathlib import Path

# Robust CUDA setup and device detection
def setup_device():
    """Set up the computation device with robust CUDA handling."""
    
    # Clear any existing CUDA context
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: CUDA cleanup failed: {e}")
    
    # Detect available device
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0]).cuda()
            device = torch.device("cuda")
            print(f"✅ Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            print(f"⚠️  CUDA available but failed initialization: {e}")
            print("   Falling back to CPU")
            return torch.device("cpu")
    else:
        print("ℹ️  CUDA not available, using CPU")
        return torch.device("cpu")

# Set up device early
DEVICE = setup_device()

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Target words for the experiment
TARGET_WORDS = {"animals", "furniture", "food"}

def filter_sentences_by_prediction_rep(sentences: List[str], target_word: str, 
                                       model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[str]:
    """
    Filter sentences to only include those where the token before target_word
    predicts target_word as the highest probability next token with confidence > 0.15.
    """
    print(f"Filtering {len(sentences)} sentences where '{target_word}' has highest next-token probability with confidence > 0.15...")
    
    filtered_sentences = []
    target_word_lower = target_word.lower()
    
    # Get target word token variations
    target_variations = [
        target_word, target_word.capitalize(), target_word.upper(),
        target_word[:-1], target_word[:-1].capitalize(), target_word[:-1].upper()  # singular forms
    ]
    target_token_ids = set()
    for variation in target_variations:
        # Try different tokenization approaches
        tokens = tokenizer.encode(f" {variation}", add_special_tokens=False)
        target_token_ids.update(tokens)
        tokens = tokenizer.encode(variation, add_special_tokens=False)
        target_token_ids.update(tokens)
    
    print(f"Target token IDs for '{target_word}': {sorted(target_token_ids)}")
    
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"  Processing sentence {i+1}/{len(sentences)}...")
        
        # Find position before target word
        pos_before_target = find_token_before_target_word(tokenizer, sentence, target_word)
        if pos_before_target is None:
            continue
        
        try:
            # Tokenize and run model
            token_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(token_ids)
                logits = outputs.logits[0, pos_before_target, :]  # Next token logits
                
                # Get probabilities and the most likely token
                probs = F.softmax(logits, dim=-1)
                top_token_id = torch.argmax(logits).item()
                top_prob = probs[top_token_id].item()
                
                # Check if the most likely token is our target word AND has confidence > 0.15
                if top_token_id in target_token_ids and top_prob > 0.15:
                    filtered_sentences.append(sentence)
                    
        except Exception as e:
            # Skip sentences that cause errors
            continue
    
    print(f"Filtered to {len(filtered_sentences)} sentences where '{target_word}' has highest probability with confidence > 0.15")
    return filtered_sentences

def load_sentences_with_target_words(data_file: str, num_sentences: Optional[int] = None,
                                   model: Optional[AutoModelForCausalLM] = None, 
                                   tokenizer: Optional[AutoTokenizer] = None) -> Dict[str, List[str]]:
    """Load sentences containing target words from the manifold dataset, filtered by prediction accuracy."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    sentences_by_target = {}
    for target_word in TARGET_WORDS:
        if target_word in data:
            all_sentences = data[target_word]
            print(f"Available sentences for '{target_word}': {len(all_sentences)}")
            
            # Filter sentences if model and tokenizer are provided
            if model is not None and tokenizer is not None:
                filtered_sentences = filter_sentences_by_prediction_rep(all_sentences, target_word, model, tokenizer)
                if num_sentences is not None:
                    if len(filtered_sentences) < num_sentences:
                        print(f"Warning: Only {len(filtered_sentences)} sentences pass filter for '{target_word}', requested {num_sentences}")
                        sentences = filtered_sentences
                    else:
                        # Select diverse sentences from filtered set
                        step = len(filtered_sentences) // num_sentences
                        sentences = [filtered_sentences[i * step] for i in range(num_sentences)]
                else:
                    sentences = filtered_sentences
            else:
                # Fallback to original selection method
                if num_sentences is not None:
                    sentences = all_sentences[:num_sentences]
                else:
                    sentences = all_sentences
            
            sentences_by_target[target_word] = sentences
            print(f"Selected {len(sentences)} sentences for '{target_word}'")
        else:
            raise ValueError(f"Target word '{target_word}' not found in data file")
    
    return sentences_by_target

def find_token_before_target_word(tokenizer: AutoTokenizer, sentence: str, target_word: str) -> Optional[int]:
    """
    Find the position of the token immediately before the target word's first token.
    Ensures we're not returning positions of spaces or special tokens.
    
    Returns the token position (not the position before target if that would be a space/special token).
    """
    # Tokenize the sentence
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    target_word_lower = target_word.lower()
    
    # Find the target word in the tokenized sequence
    for i, token in enumerate(tokens):
        # Safety check for None tokens
        if token is None:
            continue
            
        # For Llama tokenization, remove the leading space indicator ▁
        clean_token = token.replace('▁', '').lower()
        
        # Check if this token matches the target word
        if (clean_token == target_word_lower or 
            clean_token.startswith(target_word_lower) or
            (len(clean_token) > 3 and target_word_lower in clean_token)):
            
            # Find a valid previous token (not space, not special tokens)
            for j in range(i - 1, -1, -1):
                prev_token = tokens[j]
                # Safety check for None tokens
                if prev_token is None:
                    continue
                    
                prev_clean = prev_token.replace('▁', '').strip()
                
                # Skip if it's just a space marker, special token, or empty
                if (prev_clean and 
                    not prev_token.startswith('<') and 
                    not prev_token.startswith('[') and
                    len(prev_clean) > 0 and
                    prev_clean not in ['', ' ', '\n', '\t']):
                    return j
            
            # If no valid previous token found, return None
            return None
    
    # Target word not found
    return None

class RepresentationLensingExtractor:
    """Extract embeddings from specific model layers."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_pre_target_embeddings(self, sentences_by_target: Dict[str, List[str]], 
                                    layer_indices: List[int]) -> Dict[str, Dict[int, torch.Tensor]]:
        """Extract embeddings from tokens immediately before target words."""
        
        results = {}
        
        for target_word, sentences in sentences_by_target.items():
            print(f"\nProcessing target word: '{target_word}'")
            results[target_word] = {layer_idx: [] for layer_idx in layer_indices}
            
            for sentence_idx, sentence in enumerate(sentences):
                # Find position before target word
                pos_before_target = find_token_before_target_word(self.tokenizer, sentence, target_word)
                
                if pos_before_target is None:
                    continue
                
                # Tokenize and get embeddings
                token_ids = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(self.device)
                
                # Hook to capture layer outputs
                layer_outputs = {}
                
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        layer_outputs[layer_idx] = output[0]  # Get hidden states
                    return hook
                
                # Register hooks for specified layers
                hooks = []
                for layer_idx in layer_indices:
                    hook = self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)
                
                # Forward pass
                with torch.no_grad():
                    _ = self.model(token_ids)
                
                # Extract embeddings at the specified position
                for layer_idx in layer_indices:
                    if layer_idx in layer_outputs and pos_before_target < layer_outputs[layer_idx].shape[1]:
                        embedding = layer_outputs[layer_idx][0, pos_before_target, :]  # [hidden_size]
                        results[target_word][layer_idx].append(embedding)
                
                # Clean up hooks
                for hook in hooks:
                    hook.remove()
            
            # Convert lists to tensors
            for layer_idx in layer_indices:
                if results[target_word][layer_idx]:
                    results[target_word][layer_idx] = torch.stack(results[target_word][layer_idx])
                    print(f"  Layer {layer_idx}: extracted {results[target_word][layer_idx].shape[0]} embeddings")
                else:
                    # Use empty tensor for consistency
                    results[target_word][layer_idx] = torch.empty(0, self.model.config.hidden_size).to(self.device)
                    print(f"  Layer {layer_idx}: no valid embeddings extracted")
        
        return results

class LanguageModelingAnalyzer:
    """Analyze next-token predictions using either the language modeling head or embedding transpose."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.lm_head = model.lm_head
        self.embed_tokens = model.model.embed_tokens
        self.final_norm = model.model.norm  # Add reference to final RMSNorm
        self.device = next(model.parameters()).device
    
    def get_top_predictions(self, embeddings: torch.Tensor, top_k: int = 10, 
                          projection_method: str = "lm_head") -> Dict:
        """Get top-k next token predictions for a batch of embeddings and rank target words.
        
        Args:
            embeddings: Input embeddings [batch_size, hidden_size]
            top_k: Number of top predictions to return
            projection_method: Either "lm_head", "embedding_transpose", or "lm_head_with_norm"
        """
        
        # Ensure correct dtype for computation
        embeddings = embeddings.to(dtype=self.lm_head.weight.dtype)
        
        # Get logits using specified projection method
        if projection_method == "lm_head":
            logits = self.lm_head(embeddings)  # [batch_size, vocab_size]
        elif projection_method == "embedding_transpose":
            # Use transpose of embedding matrix: embeddings @ embed_weights.T
            embed_weights = self.embed_tokens.weight  # [vocab_size, hidden_size]
            logits = torch.matmul(embeddings, embed_weights.T)  # [batch_size, vocab_size]
        elif projection_method == "lm_head_with_norm":
            # Apply final RMSNorm then LM head (complete pipeline like standard forward pass)
            normalized_embeddings = self.final_norm(embeddings)
            logits = self.lm_head(normalized_embeddings)  # [batch_size, vocab_size]
        else:
            raise ValueError(f"Unknown projection_method: {projection_method}. Use 'lm_head', 'embedding_transpose', or 'lm_head_with_norm'")
        
        probs = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]
        
        # Average probabilities across all embeddings in the batch
        avg_probs = torch.mean(probs, dim=0)  # [vocab_size]
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(avg_probs, top_k)
        
        # Convert top-k to list of (token, probability)
        top_predictions = []
        for i in range(top_k):
            token_str = self.tokenizer.decode([top_k_indices[i].item()])
            # Safety check for None token strings
            if token_str is None:
                token_str = f"<UNK_{top_k_indices[i].item()}>"
            prob = top_k_probs[i].item()
            top_predictions.append((token_str, prob))
        
        # Find rankings of target category words
        target_word_rankings = {}
        
        # Sort all probabilities to get rankings
        sorted_probs, sorted_indices = torch.sort(avg_probs, descending=True)
        
        # Check each target word and its variations
        for target_word in TARGET_WORDS:
            target_variations = [
                target_word,                    # "animals"
                target_word.capitalize(),       # "Animals" 
                target_word.upper(),           # "ANIMALS"
                target_word[:-1],              # "animal" (singular)
                target_word[:-1].capitalize(), # "Animal"
                target_word[:-1].upper(),      # "ANIMAL"
            ]
            
            best_rank = None
            best_prob = 0.0
            best_token = None
            
            # Search through all tokens to find the best ranking target word variant
            for rank, token_id in enumerate(sorted_indices):
                token_str = self.tokenizer.decode([token_id.item()])
                # Safety check for None token strings
                if token_str is None:
                    continue
                token_str = token_str.strip()
                
                # Check if this token matches any target word variation
                for variation in target_variations:
                    if token_str.lower() == variation.lower() or variation.lower() in token_str.lower():
                        if best_rank is None or rank < best_rank:
                            best_rank = rank + 1  # Convert to 1-indexed ranking
                            best_prob = sorted_probs[rank].item()
                            best_token = token_str
                        break
                
                # Stop searching after reasonable number of tokens to avoid excessive computation
                if rank > 100:  # Only check top 100 tokens
                    break
            
            if best_rank is not None:
                target_word_rankings[target_word] = {
                    'rank': best_rank,
                    'probability': best_prob,
                    'token': best_token
                }
        
        return {
            "predictions": top_predictions,
            "target_word_rankings": target_word_rankings
        }
    
    def analyze_predictions_by_target(self, embeddings_by_target: Dict[str, Dict[int, torch.Tensor]], 
                                    layers: List[int], 
                                    projection_method: str = "lm_head") -> Dict[str, Dict[int, Dict]]:
        """Analyze next-token predictions for each target word and layer using specified projection method."""
        
        results = {}
        if projection_method == "lm_head":
            method_name = "LM Head"
        elif projection_method == "embedding_transpose":
            method_name = "Embedding Transpose"
        elif projection_method == "lm_head_with_norm":
            method_name = "LM Head with RMSNorm"
        else:
            method_name = projection_method.upper().replace('_', ' ')
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\nAnalyzing predictions for '{target_word}' using {method_name}:")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                
                if embeddings.numel() > 0:
                    predictions_data = self.get_top_predictions(embeddings, top_k=10, 
                                                              projection_method=projection_method)
                    results[target_word][layer_idx] = predictions_data
                    
                    print(f"  Layer {layer_idx} - Top 10 predictions:")
                    for i, (token, prob) in enumerate(predictions_data["predictions"]):
                        print(f"    {i+1}. '{token}': {prob:.6f}")
                    
                    # Display target word rankings
                    rankings = predictions_data["target_word_rankings"]
                    if rankings:
                        print(f"  Layer {layer_idx} - Target word rankings:")
                        for word, rank_info in rankings.items():
                            print(f"    '{word}': {rank_info['rank']} with probability {rank_info['probability']:.8f} (token: '{rank_info['token']}')")
                    else:
                        print(f"  Layer {layer_idx} - Target words not found in top 1000 predictions")
                else:
                    results[target_word][layer_idx] = {"predictions": [], "target_word_rankings": {}}
                    print(f"  Layer {layer_idx}: No embeddings to analyze")
        
        return results

class LocalPCAAnalyzer:
    """Perform local PCA analysis on embeddings."""
    
    def compute_local_pca(self, embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local PCA by centering embeddings on their mean."""
        
        # Convert to numpy and center the data
        embeddings_np = embeddings.cpu().numpy()
        mean_embedding = np.mean(embeddings_np, axis=0)
        centered_embeddings = embeddings_np - mean_embedding
        
        # Compute PCA - increase components to at least 100 for testing 10th, 50th, 100th PCs
        max_components = min(centered_embeddings.shape[0] - 1, centered_embeddings.shape[1], 100)
        pca = PCA(n_components=max_components)
        pca.fit(centered_embeddings)
        
        return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_
    
    def analyze_all_targets(self, embeddings_by_target: Dict[str, Dict[int, torch.Tensor]], 
                          layers: List[int]) -> Dict[str, Dict[int, Dict]]:
        """Perform PCA analysis for all target words and layers."""
        
        results = {}
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\nPCA analysis for '{target_word}':")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                
                if embeddings.numel() > 0 and embeddings.shape[0] > 1:
                    components, eigenvalues, explained_variance = self.compute_local_pca(embeddings)
                    
                    results[target_word][layer_idx] = {
                        'components': components,
                        'eigenvalues': eigenvalues,
                        'explained_variance': explained_variance
                    }
                    
                    print(f"  Layer {layer_idx}: Top PC explains {explained_variance[0]:.4f} of variance")
                else:
                    results[target_word][layer_idx] = {
                        'components': np.array([]),
                        'eigenvalues': np.array([]),
                        'explained_variance': np.array([])
                    }
                    print(f"  Layer {layer_idx}: Insufficient data for PCA")
        
        return results

class PerturbationAnalyzer:
    """Analyze effects of perturbations along principal components."""
    
    def __init__(self, lm_analyzer: LanguageModelingAnalyzer):
        self.lm_analyzer = lm_analyzer
    
    def perturb_and_analyze(self, embeddings_by_target: Dict[str, Dict[int, torch.Tensor]], 
                          pca_results: Dict[str, Dict[int, Dict]], 
                          layers: List[int], 
                          perturbation_factors: List[float] = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                          pc_indices: List[int] = [1, 2, 3, 4, 5, 10, 50],
                          projection_method: str = "lm_head",
                          include_centroid: bool = True) -> Dict:
        """Perturb embeddings along specified PCs and analyze predictions.
        
        Args:
            embeddings_by_target: Embeddings organized by target word and layer
            pca_results: PCA results for each target word and layer
            layers: Layer indices to analyze
            perturbation_factors: Scaling factors for perturbations
            pc_indices: Which principal components to test (1-indexed)
            projection_method: Either "lm_head", "embedding_transpose", or "lm_head_with_norm"
            include_centroid: Whether to include centroid-based perturbations
        """
        
        results = {}
        if projection_method == "lm_head":
            method_name = "LM Head"
        elif projection_method == "embedding_transpose":
            method_name = "Embedding Transpose"
        elif projection_method == "lm_head_with_norm":
            method_name = "LM Head with RMSNorm"
        else:
            method_name = projection_method.upper().replace('_', ' ')
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\nPerturbation analysis for '{target_word}' using {method_name}:")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                pca_data = pca_results[target_word][layer_idx]
                
                if embeddings.numel() > 0 and len(pca_data['components']) > 0:
                    results[target_word][layer_idx] = {}
                    
                    # Convert embeddings to numpy once for all perturbations
                    embeddings_np = embeddings.cpu().numpy()
                    
                    # Test perturbations along specified principal components
                    for pc_idx in pc_indices:
                        pc_array_idx = pc_idx - 1  # Convert to 0-indexed
                        
                        # Check if this PC exists
                        if pc_array_idx >= len(pca_data['components']):
                            print(f"  Layer {layer_idx}: PC {pc_idx} not available (only {len(pca_data['components'])} components)")
                            continue
                        
                        # Get the specified principal component
                        pc = pca_data['components'][pc_array_idx]
                        eigenvalue = pca_data['eigenvalues'][pc_array_idx]
                        explained_var = pca_data['explained_variance'][pc_array_idx]
                        
                        print(f"  Layer {layer_idx}, PC {pc_idx} (explains {explained_var:.4f} of variance):")
                        results[target_word][layer_idx][f'PC_{pc_idx}'] = {}
                        
                        for factor in perturbation_factors:
                            # Perturb embeddings
                            perturbation = eigenvalue * factor * pc
                            perturbed_embeddings_np = embeddings_np + perturbation
                            
                            # Convert back to tensor with correct dtype
                            perturbed_tensor = torch.from_numpy(perturbed_embeddings_np).to(
                                device=embeddings.device, dtype=embeddings.dtype)
                            
                            # Analyze predictions using specified projection method
                            predictions_data = self.lm_analyzer.get_top_predictions(perturbed_tensor, top_k=10,
                                                                                   projection_method=projection_method)
                            results[target_word][layer_idx][f'PC_{pc_idx}'][factor] = predictions_data
                            
                            print(f"    Factor {factor} - Top 5 predictions:")
                            for i, (token, prob) in enumerate(predictions_data["predictions"][:5]):
                                print(f"      {i+1}. '{token}': {prob:.6f}")
                            
                            # Display target word rankings for perturbations
                            rankings = predictions_data["target_word_rankings"]
                            if rankings:
                                print(f"    Factor {factor} - Target word rankings:")
                                for word, rank_info in rankings.items():
                                    print(f"      '{word}': {rank_info['rank']} (prob: {rank_info['probability']:.8f})")
                            else:
                                print(f"    Factor {factor} - Target words not in top 100 predictions")
                            print()  # Add spacing between perturbation factors
                    
                    # Add centroid-based perturbations using all three category centroids
                    if include_centroid:
                        # First, compute centroids for all target words at this layer
                        all_centroids = {}
                        for target in embeddings_by_target.keys():
                            target_embeddings = embeddings_by_target[target][layer_idx]
                            if target_embeddings.numel() > 0:
                                target_embeddings_np = target_embeddings.cpu().numpy()
                                target_centroid = np.mean(target_embeddings_np, axis=0)
                                all_centroids[target] = target_centroid
                        
                        # Now perturb current target's embeddings using all three centroids
                        for centroid_source, centroid in all_centroids.items():
                            # Normalize centroid to unit vector (direction only)
                            centroid_direction = centroid / (np.linalg.norm(centroid) + 1e-8)
                            
                            # Use magnitude of the centroid position vector
                            centroid_magnitude = np.linalg.norm(centroid)
                            
                            print(f"  Layer {layer_idx}, {centroid_source} centroid perturbation (magnitude: {centroid_magnitude:.6f}):")
                            results[target_word][layer_idx][f'Centroid_{centroid_source}'] = {}
                            
                            for factor in perturbation_factors:
                                # Perturb embeddings using centroid direction with centroid magnitude
                                perturbation = centroid_magnitude * factor * centroid_direction
                                perturbed_embeddings_np = embeddings_np + perturbation
                                
                                # Convert back to tensor with correct dtype
                                perturbed_tensor = torch.from_numpy(perturbed_embeddings_np).to(
                                    device=embeddings.device, dtype=embeddings.dtype)
                                
                                # Analyze predictions using specified projection method
                                predictions_data = self.lm_analyzer.get_top_predictions(perturbed_tensor, top_k=10,
                                                                                       projection_method=projection_method)
                                results[target_word][layer_idx][f'Centroid_{centroid_source}'][factor] = predictions_data
                                
                                print(f"    Factor {factor} - Top 5 predictions:")
                                for i, (token, prob) in enumerate(predictions_data["predictions"][:5]):
                                    print(f"      {i+1}. '{token}': {prob:.6f}")
                                
                                # Display target word rankings for perturbations
                                rankings = predictions_data["target_word_rankings"]
                                if rankings:
                                    print(f"    Factor {factor} - Target word rankings:")
                                    for word, rank_info in rankings.items():
                                        print(f"      '{word}': {rank_info['rank']} (prob: {rank_info['probability']:.8f})")
                                else:
                                    print(f"    Factor {factor} - Target words not in top 100 predictions")
                                print()  # Add spacing between perturbation factors
                else:
                    results[target_word][layer_idx] = {}
                    print(f"  Layer {layer_idx}: No data for perturbation")
        
        return results

class AdditionalTestCaseAnalyzer:
    """Analyze additional test cases using pre-computed PCs from the original experiment."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, lm_analyzer: LanguageModelingAnalyzer):
        self.model = model
        self.tokenizer = tokenizer
        self.lm_analyzer = lm_analyzer
        self.device = next(model.parameters()).device
    
    def create_test_cases(self) -> Dict[str, List[str]]:
        """Create the three types of test cases for each target word."""
        test_cases = {}
        
        # German translations
        german_words = {
            "animals": "Tiere",
            "furniture": "Möbel", 
            "food": "Essen"
        }
        
        for target_word in TARGET_WORDS:
            test_cases[target_word] = []
            
            # Type 1: Just the category word
            test_cases[target_word].append(target_word)
            
            # Type 2: Spelling format
            spelled_word = "-".join(target_word.lower())
            test_cases[target_word].append(f"{spelled_word} spells ")
            
            # Type 3: Translation format
            german_word = german_words[target_word]
            test_cases[target_word].append(f"Translated from German into English, '{german_word}' is ")
        
        return test_cases
    
    def extract_last_token_embeddings(self, test_cases: Dict[str, List[str]], 
                                    layers: List[int]) -> Dict[str, Dict[int, Dict[int, torch.Tensor]]]:
        """Extract embeddings from the last token position for each test case."""
        
        results = {}
        
        for target_word, cases in test_cases.items():
            print(f"\nExtracting embeddings for '{target_word}' test cases:")
            results[target_word] = {layer_idx: {} for layer_idx in layers}
            
            for case_idx, test_case in enumerate(cases):
                case_types = ["category_word", "spelling", "translation"]
                case_type = case_types[case_idx]
                print(f"  {case_type}: '{test_case}'")
                
                # Tokenize the test case
                token_ids = self.tokenizer.encode(test_case, add_special_tokens=True, return_tensors="pt").to(self.device)
                last_token_pos = token_ids.shape[1] - 1  # Last token position
                
                # Hook to capture layer outputs
                layer_outputs = {}
                
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        layer_outputs[layer_idx] = output[0]  # Get hidden states
                    return hook
                
                # Register hooks for specified layers
                hooks = []
                for layer_idx in layers:
                    hook = self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                    hooks.append(hook)
                
                # Forward pass
                with torch.no_grad():
                    _ = self.model(token_ids)
                
                # Extract embeddings at the last token position
                for layer_idx in layers:
                    if layer_idx in layer_outputs and last_token_pos < layer_outputs[layer_idx].shape[1]:
                        embedding = layer_outputs[layer_idx][0, last_token_pos, :]  # [hidden_size]
                        results[target_word][layer_idx][case_idx] = embedding
                
                # Clean up hooks
                for hook in hooks:
                    hook.remove()
        
        return results
    
    def analyze_test_cases_with_pcs(self, test_case_embeddings: Dict[str, Dict[int, Dict[int, torch.Tensor]]], 
                                  pca_results: Dict[str, Dict[int, Dict]], 
                                  layers: List[int],
                                  embeddings_by_target: Dict[str, Dict[int, torch.Tensor]] = None,
                                  perturbation_factors: List[float] = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                                  pc_indices: List[int] = [1, 2, 3, 4, 5, 10, 50],
                                  projection_method: str = "lm_head",
                                  include_centroid: bool = True) -> Dict:
        """Analyze test cases using pre-computed PCs from the original experiment.
        
        Args:
            test_case_embeddings: Embeddings from test cases
            pca_results: Pre-computed PCA results from original experiment  
            layers: Layer indices to analyze
            embeddings_by_target: Original experiment embeddings (needed for centroid computation)
            perturbation_factors: Scaling factors for perturbations
            pc_indices: Which principal components to test (1-indexed)
            projection_method: Either "lm_head", "embedding_transpose", or "lm_head_with_norm"
            include_centroid: Whether to include centroid-based perturbations
        """
        
        results = {}
        case_types = ["category_word", "spelling", "translation"]
        if projection_method == "lm_head":
            method_name = "LM Head"
        elif projection_method == "embedding_transpose":
            method_name = "Embedding Transpose"
        elif projection_method == "lm_head_with_norm":
            method_name = "LM Head with RMSNorm"
        else:
            method_name = projection_method.upper().replace('_', ' ')
        
        for target_word in test_case_embeddings.keys():
            results[target_word] = {}
            print(f"\nAnalyzing test cases for '{target_word}' using pre-computed PCs and {method_name}:")
            
            for layer_idx in layers:
                results[target_word][layer_idx] = {}
                pca_data = pca_results[target_word][layer_idx]
                
                if len(pca_data['components']) == 0:
                    print(f"  Layer {layer_idx}: No PCA data available")
                    continue
                
                print(f"  Layer {layer_idx}:")
                
                for case_idx, case_type in enumerate(case_types):
                    if case_idx not in test_case_embeddings[target_word][layer_idx]:
                        continue
                    
                    embedding = test_case_embeddings[target_word][layer_idx][case_idx]
                    results[target_word][layer_idx][case_type] = {}
                    
                    print(f"    {case_type}:")
                    
                    # Get baseline predictions (no perturbation)
                    baseline_predictions = self.lm_analyzer.get_top_predictions(embedding.unsqueeze(0), top_k=10,
                                                                              projection_method=projection_method)
                    results[target_word][layer_idx][case_type]['baseline'] = baseline_predictions
                    
                    print(f"      Baseline - Top 5 predictions:")
                    for i, (token, prob) in enumerate(baseline_predictions["predictions"][:5]):
                        print(f"        {i+1}. '{token}': {prob:.6f}")
                    
                    # Test perturbations along specified principal components
                    for pc_idx in pc_indices:
                        pc_array_idx = pc_idx - 1  # Convert to 0-indexed
                        
                        # Check if this PC exists
                        if pc_array_idx >= len(pca_data['components']):
                            continue
                        
                        # Get the specified principal component from original experiment
                        pc = pca_data['components'][pc_array_idx]
                        eigenvalue = pca_data['eigenvalues'][pc_array_idx]
                        
                        results[target_word][layer_idx][case_type][f'PC_{pc_idx}'] = {}
                        
                        for factor in perturbation_factors:
                            # Apply perturbation
                            perturbation = eigenvalue * factor * pc
                            embedding_np = embedding.cpu().numpy()
                            perturbed_embedding_np = embedding_np + perturbation
                            
                            # Convert back to tensor
                            perturbed_tensor = torch.from_numpy(perturbed_embedding_np).to(
                                device=embedding.device, dtype=embedding.dtype).unsqueeze(0)
                            
                            # Analyze predictions using specified projection method
                            predictions_data = self.lm_analyzer.get_top_predictions(perturbed_tensor, top_k=10,
                                                                                   projection_method=projection_method)
                            results[target_word][layer_idx][case_type][f'PC_{pc_idx}'][factor] = predictions_data
                            
                            # Print top predictions for first few factors
                            if factor in [-1.0, 1.0]:  # Only print for key factors to avoid clutter
                                print(f"      PC{pc_idx}, factor {factor} - Top 3 predictions:")
                                for i, (token, prob) in enumerate(predictions_data["predictions"][:3]):
                                    print(f"        {i+1}. '{token}': {prob:.6f}")
                    
                    # Add centroid-based perturbations for test cases using all three category centroids
                    if include_centroid:
                        # Compute centroids for all target words at this layer from original experiment
                        all_centroids = {}
                        for target in embeddings_by_target.keys():
                            target_embeddings = embeddings_by_target.get(target, {}).get(layer_idx)
                            if target_embeddings is not None and target_embeddings.numel() > 0:
                                target_embeddings_np = target_embeddings.cpu().numpy()
                                target_centroid = np.mean(target_embeddings_np, axis=0)
                                all_centroids[target] = target_centroid
                        
                        # Apply perturbations using all three centroids
                        for centroid_source, centroid in all_centroids.items():
                            # Normalize centroid to unit vector (direction only)
                            centroid_direction = centroid / (np.linalg.norm(centroid) + 1e-8)
                            
                            # Use magnitude of the centroid position vector
                            centroid_magnitude = np.linalg.norm(centroid)
                            
                            results[target_word][layer_idx][case_type][f'Centroid_{centroid_source}'] = {}
                            
                            for factor in perturbation_factors:
                                # Apply centroid perturbation to the test case embedding
                                perturbation = centroid_magnitude * factor * centroid_direction
                                embedding_np = embedding.cpu().numpy()
                                perturbed_embedding_np = embedding_np + perturbation
                                
                                # Convert back to tensor
                                perturbed_tensor = torch.from_numpy(perturbed_embedding_np).to(
                                    device=embedding.device, dtype=embedding.dtype).unsqueeze(0)
                                
                                # Analyze predictions using specified projection method
                                predictions_data = self.lm_analyzer.get_top_predictions(perturbed_tensor, top_k=10,
                                                                                       projection_method=projection_method)
                                results[target_word][layer_idx][case_type][f'Centroid_{centroid_source}'][factor] = predictions_data
                                
                                # Print top predictions for key factors
                                if factor in [-1.0, 1.0]:  # Only print for key factors to avoid clutter
                                    print(f"      {centroid_source} Centroid, factor {factor} - Top 3 predictions:")
                                    for i, (token, prob) in enumerate(predictions_data["predictions"][:3]):
                                        print(f"        {i+1}. '{token}': {prob:.6f}")
        
        return results

class Layer31DualAnalyzer:
    """Special analyzer for layer 31 that compares standard vs extracted embedding analysis."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, lm_analyzer: LanguageModelingAnalyzer):
        self.model = model
        self.tokenizer = tokenizer
        self.lm_analyzer = lm_analyzer
        self.device = next(model.parameters()).device
    
    def analyze_layer31_dual(self, sentences_by_target: Dict[str, List[str]]) -> Dict:
        """Analyze layer 31 using both standard and extracted methods for LM head projection only."""
        
        results = {}
        
        for target_word, sentences in sentences_by_target.items():
            print(f"\nLayer 31 dual analysis for '{target_word}':")
            results[target_word] = {
                'standard_method': {},  # Normal forward pass
                'extracted_method': {} # Extracted embeddings like other layers
            }
            
            # For each sentence, we'll get predictions using both methods
            standard_embeddings = []
            extracted_embeddings = []
            
            for sentence_idx, sentence in enumerate(sentences):
                # Find position before target word
                pos_before_target = find_token_before_target_word(self.tokenizer, sentence, target_word)
                if pos_before_target is None:
                    continue
                
                # Tokenize
                token_ids = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(self.device)
                
                # Method 1: Standard forward pass (let model complete naturally)
                with torch.no_grad():
                    standard_outputs = self.model(token_ids)
                    standard_logits = standard_outputs.logits[0, pos_before_target, :]  # Get logits directly
                    standard_probs = F.softmax(standard_logits, dim=-1)
                    standard_embeddings.append(standard_probs)  # Store the probability distribution
                
                # Method 2: Extract layer 31 embeddings and pass to LM head
                layer_31_embedding = None
                
                def hook_fn(module, input, output):
                    nonlocal layer_31_embedding
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    if pos_before_target < hidden_states.shape[1]:
                        layer_31_embedding = hidden_states[0, pos_before_target, :].detach().clone()
                
                # Hook layer 31
                hook = self.model.model.layers[31].register_forward_hook(hook_fn)
                
                try:
                    with torch.no_grad():
                        _ = self.model(token_ids)
                    
                    if layer_31_embedding is not None:
                        # Pass directly to LM head (bypassing any final layers)
                        extracted_logits = self.model.lm_head(layer_31_embedding)
                        extracted_probs = F.softmax(extracted_logits, dim=-1)
                        extracted_embeddings.append(extracted_probs)
                
                finally:
                    hook.remove()
            
            if standard_embeddings and extracted_embeddings:
                # Convert to tensors and analyze
                standard_tensor = torch.stack(standard_embeddings)  # [num_sentences, vocab_size]
                extracted_tensor = torch.stack(extracted_embeddings)  # [num_sentences, vocab_size]
                
                # Get average probability distributions
                standard_avg_probs = torch.mean(standard_tensor, dim=0)  # [vocab_size]
                extracted_avg_probs = torch.mean(extracted_tensor, dim=0)  # [vocab_size]
                
                # Get top predictions for both methods
                standard_top_probs, standard_top_indices = torch.topk(standard_avg_probs, 10)
                extracted_top_probs, extracted_top_indices = torch.topk(extracted_avg_probs, 10)
                
                # Format results
                results[target_word]['standard_method'] = {
                    'predictions': [(self.tokenizer.decode([idx.item()]) or f"<UNK_{idx.item()}>", prob.item()) 
                                  for idx, prob in zip(standard_top_indices, standard_top_probs)],
                    'avg_probs': standard_avg_probs
                }
                
                results[target_word]['extracted_method'] = {
                    'predictions': [(self.tokenizer.decode([idx.item()]) or f"<UNK_{idx.item()}>", prob.item()) 
                                  for idx, prob in zip(extracted_top_indices, extracted_top_probs)],
                    'avg_probs': extracted_avg_probs
                }
                
                # Print comparison
                print(f"  Standard method - Top 5 predictions:")
                for i, (token, prob) in enumerate(results[target_word]['standard_method']['predictions'][:5]):
                    print(f"    {i+1}. '{token}': {prob:.6f}")
                
                print(f"  Extracted method - Top 5 predictions:")
                for i, (token, prob) in enumerate(results[target_word]['extracted_method']['predictions'][:5]):
                    print(f"    {i+1}. '{token}': {prob:.6f}")
                
                # Compute similarity between methods
                cosine_sim = F.cosine_similarity(standard_avg_probs.unsqueeze(0), 
                                               extracted_avg_probs.unsqueeze(0))
                print(f"  Cosine similarity between methods: {cosine_sim.item():.6f}")
        
        return results

def create_cosine_similarity_matrix(pca_results: Dict[str, Dict[int, Dict]], 
                                  layers: List[int], 
                                  target_words: List[str],
                                  pc_index: int = 1) -> np.ndarray:
    """Create cosine similarity matrix between specified PCs.
    
    Args:
        pca_results: PCA results for all targets and layers
        layers: Layer indices to include
        target_words: Target words to include
        pc_index: Which principal component to use (1-indexed, default=1 for top PC)
    """
    
    # Collect all valid PCs
    all_pcs = []
    labels = []
    pc_array_idx = pc_index - 1  # Convert to 0-indexed
    
    for target_word in target_words:
        for layer_idx in layers:
            if (target_word in pca_results and 
                layer_idx in pca_results[target_word] and 
                len(pca_results[target_word][layer_idx]['components']) > pc_array_idx):
                
                pc = pca_results[target_word][layer_idx]['components'][pc_array_idx]  # Specified PC
                all_pcs.append(pc)
                labels.append(f"{target_word}_L{layer_idx}_PC{pc_index}")
    
    if not all_pcs:
        return np.array([]), []
    
    # Compute cosine similarity matrix
    all_pcs = np.array(all_pcs)
    similarity_matrix = np.zeros((len(all_pcs), len(all_pcs)))
    
    for i in range(len(all_pcs)):
        for j in range(len(all_pcs)):
            # Cosine similarity
            dot_product = np.dot(all_pcs[i], all_pcs[j])
            norm_i = np.linalg.norm(all_pcs[i])
            norm_j = np.linalg.norm(all_pcs[j])
            similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
    
    return similarity_matrix, labels

def plot_cosine_similarity_matrix(similarity_matrix: np.ndarray, labels: List[str], output_path: str):
    """Plot the cosine similarity matrix with 2 decimal place precision."""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='.2f',  # Format annotations to 2 decimal places
                cmap='coolwarm', 
                center=0,
                square=True)
    plt.title('Cosine Similarity Between Principal Components')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_representation_lensing_experiment(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    data_file: str = "../tools/manifold_sentences_hard_exactword_1000.json",
    num_sentences: Optional[int] = None,
    layers: List[int] = [0, 5, 10, 15, 20, 25, 30, 31],
    output_dir: str = "../rep_lensing/lens_results",
    projection_methods: List[str] = ["lm_head", "embedding_transpose", "lm_head_with_norm"]
):
    """Run the complete representation lensing experiment."""
    
    print("=" * 60)
    print("REPRESENTATION LENSING EXPERIMENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # 1. Load model and tokenizer
    print("\n1. Loading Llama 3.1-8B model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with device-appropriate settings
        if DEVICE.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # CPU fallback with float32
            print("   cuda failed: Loading model for CPU")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
            model = model.to(DEVICE)
        
        model.eval()
        print(f"   Model loaded successfully on {model.device}")
        
    except Exception as e:
        print(f" Error loading model: {e}")
        print("   This might be due to:")
        print("   1. Insufficient GPU memory")
        print("   2. CUDA environment issues")
        print("   3. Model not available")
        raise
    
    # 2. Load sentences (with filtering based on prediction accuracy)
    print("\n2. Loading and filtering sentences...")
    sentences_by_target = load_sentences_with_target_words(data_file, num_sentences, model, tokenizer)
    
    # 3. Extract embeddings
    print("\n3. Extracting pre-target embeddings...")
    extractor = RepresentationLensingExtractor(model, tokenizer)
    embeddings_by_target = extractor.extract_pre_target_embeddings(sentences_by_target, layers)
    
    # 4. Analyze baseline predictions using specified projection methods
    print("\n4. Analyzing baseline next-token predictions...")
    lm_analyzer = LanguageModelingAnalyzer(model, tokenizer)
    
    # Run analysis with specified projection methods
    baseline_predictions = {}
    perturbation_results = {}
    test_case_results = {}
    
    for projection_method in projection_methods:
        print(f"\n--- Analysis using {projection_method.upper().replace('_', ' ')} ---")
        
        baseline_predictions[projection_method] = lm_analyzer.analyze_predictions_by_target(
            embeddings_by_target, layers, projection_method=projection_method
        )
    
    # 5. Perform PCA analysis (only needs to be done once)
    print("\n5. Performing local PCA analysis...")
    pca_analyzer = LocalPCAAnalyzer()
    pca_results = pca_analyzer.analyze_all_targets(embeddings_by_target, layers)
    
    # 6. Perturbation analysis for both projection methods
    print("\n6. Running perturbation analysis...")
    perturbation_analyzer = PerturbationAnalyzer(lm_analyzer)
    
    for projection_method in projection_methods:
        print(f"\n--- Perturbation analysis using {projection_method.upper().replace('_', ' ')} ---")
        perturbation_results[projection_method] = perturbation_analyzer.perturb_and_analyze(
            embeddings_by_target, pca_results, layers, projection_method=projection_method,
            include_centroid=True
        )
    
    # 6b. Layer 31 dual analysis (LM head only)
    layer31_dual_results = {}
    if 31 in layers and "lm_head" in projection_methods:
        print("\n6b. Running Layer 31 dual analysis (standard vs extracted embeddings)...")
        layer31_analyzer = Layer31DualAnalyzer(model, tokenizer, lm_analyzer)
        layer31_dual_results = layer31_analyzer.analyze_layer31_dual(sentences_by_target)
    
    # 6c. Additional test case analysis for both projection methods
    print("\n6c. Running additional test case analysis...")
    test_case_analyzer = AdditionalTestCaseAnalyzer(model, tokenizer, lm_analyzer)
    
    # Create test cases
    test_cases = test_case_analyzer.create_test_cases()
    print("Test cases created:")
    for target_word, cases in test_cases.items():
        print(f"  {target_word}:")
        for i, case in enumerate(cases):
            case_types = ["category_word", "spelling", "translation"]
            print(f"    {case_types[i]}: '{case}'")
    
    # Extract embeddings from test cases (only needs to be done once)
    test_case_embeddings = test_case_analyzer.extract_last_token_embeddings(test_cases, layers)
    
    # Analyze test cases using pre-computed PCs for both projection methods
    for projection_method in projection_methods:
        print(f"\n--- Test case analysis using {projection_method.upper().replace('_', ' ')} ---")
        test_case_results[projection_method] = test_case_analyzer.analyze_test_cases_with_pcs(
            test_case_embeddings, pca_results, layers, 
            embeddings_by_target=embeddings_by_target,
            projection_method=projection_method
        )
    
    # 7. Create cosine similarity matrices for tested PCs
    print("\n7. Creating cosine similarity matrices...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create similarity matrices for the specific PCs we tested
    pc_indices_to_plot = [1, 2, 3, 4, 5, 10, 50]
    
    for pc_idx in pc_indices_to_plot:
        similarity_matrix, labels = create_cosine_similarity_matrix(
            pca_results, layers, list(TARGET_WORDS), pc_idx
        )
        
        if similarity_matrix.size > 0:
            plot_cosine_similarity_matrix(
                similarity_matrix, labels, 
                f"{output_dir}/pc_similarity_matrix_PC{pc_idx}.png"
            )
            print(f"   Saved cosine similarity matrix for PC {pc_idx} to {output_dir}/pc_similarity_matrix_PC{pc_idx}.png")
    
    # 8. Save results
    print("\n8. Saving results...")
    results = {
        'experiment_config': {
            'model_name': model_name,
            'target_words': list(TARGET_WORDS),
            'num_sentences': num_sentences if num_sentences else 'all_available',
            'layers': layers,
            'confidence_threshold': 0.15,
            'projection_methods': projection_methods
        },
        'baseline_predictions': baseline_predictions,
        'pca_results': {
            target: {
                layer: {
                    'explained_variance': data['explained_variance'].tolist() if len(data['explained_variance']) > 0 else [],
                    'top_variance': data['explained_variance'][0] if len(data['explained_variance']) > 0 else 0.0
                }
                for layer, data in layers_data.items()
            }
            for target, layers_data in pca_results.items()
        },
        'perturbation_results': perturbation_results,
        'test_cases': {
            target_word: cases for target_word, cases in test_cases.items()
        },
        'test_case_results': test_case_results,
        'layer31_dual_results': layer31_dual_results
    }
    
    with open(f"{output_dir}/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   Results saved to {output_dir}/experiment_results.json")
    print("\n  Experiment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run representation lensing experiment")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B", 
                       help="Model name")
    parser.add_argument("--data-file", default="../tools/manifold_sentences_hard_exactword_1000.json",
                       help="Path to sentences data file")
    parser.add_argument("--num-sentences", type=int, default=200,
                       help="Number of sentences per target (default: all)")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30, 31],
                       help="Layer indices to analyze")
    parser.add_argument("--output-dir", default="../rep_lensing/lens_results",
                       help="Output directory")
    parser.add_argument("--projection-methods", nargs="+", default=["lm_head", "embedding_transpose", "lm_head_with_norm"],
                       choices=["lm_head", "embedding_transpose", "lm_head_with_norm"],
                       help="Projection methods to use: lm_head (standard), embedding_transpose, and/or lm_head_with_norm (with final RMSNorm)")
    
    args = parser.parse_args()
    
    run_representation_lensing_experiment(
        model_name=args.model_name,
        data_file=args.data_file,
        num_sentences=args.num_sentences,
        layers=args.layers,
        output_dir=args.output_dir,
        projection_methods=args.projection_methods
    )