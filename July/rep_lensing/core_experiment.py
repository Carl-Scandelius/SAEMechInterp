#!/usr/bin/env python3
"""
Representation Lensing Experiment for Llama 3.1-8B (non-instruct)

This experiment:
1. Extracts embeddings from tokens immediately before target words
2. Analyzes next-token predictions using the language modeling head
3. Performs local PCA on embeddings and tests perturbations
4. Generates cosine similarity matrices and prediction analysis
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
from pathlib import Path

# Target words for the experiment
TARGET_WORDS = {"animals", "furniture", "food"}

def load_sentences_with_target_words(data_file: str, num_sentences: Optional[int] = None) -> Dict[str, List[str]]:
    """Load sentences containing target words from the manifold dataset."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    sentences_by_target = {}
    for target_word in TARGET_WORDS:
        if target_word in data:
            sentences = data[target_word]
            if num_sentences is not None:
                sentences = sentences[:num_sentences]
            sentences_by_target[target_word] = sentences
            print(f"Loaded {len(sentences)} sentences for '{target_word}'")
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
        # For Llama tokenization, remove the leading space indicator ▁
        clean_token = token.replace('▁', '').lower()
        
        # Check if this token matches the target word
        if (clean_token == target_word_lower or 
            clean_token.startswith(target_word_lower) or
            (len(clean_token) > 3 and target_word_lower in clean_token)):
            
            # Find a valid previous token (not space, not special tokens)
            for j in range(i - 1, -1, -1):
                prev_token = tokens[j]
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
    """Analyze next-token predictions using the language modeling head."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.lm_head = model.lm_head
        self.device = next(model.parameters()).device
    
    def get_top_predictions(self, embeddings: torch.Tensor, top_k: int = 10) -> Dict:
        """Get top-k next token predictions for a batch of embeddings and rank target words."""
        
        # Ensure correct dtype for the language modeling head
        embeddings = embeddings.to(dtype=self.lm_head.weight.dtype)
        
        # Get logits and apply softmax
        logits = self.lm_head(embeddings)  # [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1)  # [batch_size, vocab_size]
        
        # Average probabilities across all embeddings in the batch
        avg_probs = torch.mean(probs, dim=0)  # [vocab_size]
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(avg_probs, top_k)
        
        # Convert top-k to list of (token, probability)
        top_predictions = []
        for i in range(top_k):
            token_str = self.tokenizer.decode([top_k_indices[i].item()])
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
                token_str = self.tokenizer.decode([token_id.item()]).strip()
                
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
                                    layers: List[int]) -> Dict[str, Dict[int, Dict]]:
        """Analyze next-token predictions for each target word and layer."""
        
        results = {}
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\nAnalyzing predictions for '{target_word}':")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                
                if embeddings.numel() > 0:
                    predictions_data = self.get_top_predictions(embeddings, top_k=10)
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
        
        # Compute PCA
        n_components = min(centered_embeddings.shape[0] - 1, centered_embeddings.shape[1], 10)
        pca = PCA(n_components=n_components)
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
                          perturbation_factors: List[float] = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]) -> Dict:
        """Perturb embeddings along top PC and analyze predictions."""
        
        results = {}
        
        for target_word in embeddings_by_target.keys():
            results[target_word] = {}
            print(f"\nPerturbation analysis for '{target_word}':")
            
            for layer_idx in layers:
                embeddings = embeddings_by_target[target_word][layer_idx]
                pca_data = pca_results[target_word][layer_idx]
                
                if embeddings.numel() > 0 and len(pca_data['components']) > 0:
                    results[target_word][layer_idx] = {}
                    
                    # Get the top principal component
                    top_pc = pca_data['components'][0]  # First PC
                    top_eigenvalue = pca_data['eigenvalues'][0]
                    
                    for factor in perturbation_factors:
                        # Perturb embeddings
                        perturbation = top_eigenvalue * factor * top_pc
                        embeddings_np = embeddings.cpu().numpy()
                        perturbed_embeddings_np = embeddings_np + perturbation
                        
                        # Convert back to tensor with correct dtype
                        perturbed_tensor = torch.from_numpy(perturbed_embeddings_np).to(
                            device=embeddings.device, dtype=embeddings.dtype)
                        
                        # Analyze predictions
                        predictions_data = self.lm_analyzer.get_top_predictions(perturbed_tensor, top_k=10)
                        results[target_word][layer_idx][factor] = predictions_data
                        
                        print(f"  Layer {layer_idx}, factor {factor} - Top 10 predictions:")
                        for i, (token, prob) in enumerate(predictions_data["predictions"]):
                            print(f"    {i+1}. '{token}': {prob:.6f}")
                        
                        # Display target word rankings for perturbations
                        rankings = predictions_data["target_word_rankings"]
                        if rankings:
                            print(f"  Layer {layer_idx}, factor {factor} - Target word rankings:")
                            for word, rank_info in rankings.items():
                                print(f"    '{word}': {rank_info['rank']} with probability {rank_info['probability']:.8f} (token: '{rank_info['token']}')")
                        else:
                            print(f"  Layer {layer_idx}, factor {factor} - Target words not found in top 1000 predictions")
                        print()  # Add spacing between perturbation factors
                else:
                    results[target_word][layer_idx] = {}
                    print(f"  Layer {layer_idx}: No data for perturbation")
        
        return results

def create_cosine_similarity_matrix(pca_results: Dict[str, Dict[int, Dict]], 
                                  layers: List[int], 
                                  target_words: List[str]) -> np.ndarray:
    """Create cosine similarity matrix between all PCs."""
    
    # Collect all valid PCs
    all_pcs = []
    labels = []
    
    for target_word in target_words:
        for layer_idx in layers:
            if (target_word in pca_results and 
                layer_idx in pca_results[target_word] and 
                len(pca_results[target_word][layer_idx]['components']) > 0):
                
                pc = pca_results[target_word][layer_idx]['components'][0]  # Top PC
                all_pcs.append(pc)
                labels.append(f"{target_word}_L{layer_idx}")
    
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
    """Plot the cosine similarity matrix."""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
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
    layers: List[int] = [0, 5, 15, 25, 31],
    output_dir: str = "../prediction_steering_results"
):
    """Run the complete representation lensing experiment."""
    
    print("=" * 60)
    print("REPRESENTATION LENSING EXPERIMENT")
    print("=" * 60)
    
    # 1. Load model and tokenizer
    print("\n1. Loading Llama 3.1-8B model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # 2. Load sentences
    print("\n2. Loading sentences...")
    sentences_by_target = load_sentences_with_target_words(data_file, num_sentences)
    
    # 3. Extract embeddings
    print("\n3. Extracting pre-target embeddings...")
    extractor = RepresentationLensingExtractor(model, tokenizer)
    embeddings_by_target = extractor.extract_pre_target_embeddings(sentences_by_target, layers)
    
    # 4. Analyze baseline predictions
    print("\n4. Analyzing baseline next-token predictions...")
    lm_analyzer = LanguageModelingAnalyzer(model, tokenizer)
    baseline_predictions = lm_analyzer.analyze_predictions_by_target(embeddings_by_target, layers)
    
    # 5. Perform PCA analysis
    print("\n5. Performing local PCA analysis...")
    pca_analyzer = LocalPCAAnalyzer()
    pca_results = pca_analyzer.analyze_all_targets(embeddings_by_target, layers)
    
    # 6. Perturbation analysis
    print("\n6. Running perturbation analysis...")
    perturbation_analyzer = PerturbationAnalyzer(lm_analyzer)
    perturbation_results = perturbation_analyzer.perturb_and_analyze(
        embeddings_by_target, pca_results, layers
    )
    
    # 7. Create cosine similarity matrix
    print("\n7. Creating cosine similarity matrix...")
    Path(output_dir).mkdir(exist_ok=True)
    
    similarity_matrix, labels = create_cosine_similarity_matrix(
        pca_results, layers, list(TARGET_WORDS)
    )
    
    if similarity_matrix.size > 0:
        plot_cosine_similarity_matrix(
            similarity_matrix, labels, 
            f"{output_dir}/pc_similarity_matrix.png"
        )
        print(f"   Saved cosine similarity matrix to {output_dir}/pc_similarity_matrix.png")
    
    # 8. Save results
    print("\n8. Saving results...")
    results = {
        'experiment_config': {
            'model_name': model_name,
            'target_words': list(TARGET_WORDS),
            'num_sentences': num_sentences if num_sentences else 'all_available',
            'layers': layers
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
        'perturbation_results': perturbation_results
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
    parser.add_argument("--num-sentences", type=int, default=None,
                       help="Number of sentences per target (default: all)")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 5, 15, 25, 27, 30, 31],
                       help="Layer indices to analyze")
    parser.add_argument("--output-dir", default="../rep_lensing/lens_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    run_representation_lensing_experiment(
        model_name=args.model_name,
        data_file=args.data_file,
        num_sentences=args.num_sentences,
        layers=args.layers,
        output_dir=args.output_dir
    ) 