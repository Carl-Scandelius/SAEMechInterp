#!/usr/bin/env python
"""
Fisher-guided Embedding Perturbations Experiment for Llama 3.1-8B

This experiment:
1. Extracts residual stream activations h before target word "animals"
2. Computes Fisher Information Matrix I(h) = Σ_c p(c|h)[∇_h log p(c|h) ∇_h log p(c|h)^T]
3. Finds top eigenvector v and eigenvalue λ of I(h)
4. Perturbs h' = h + α*λ*v for multiple scales α
5. Analyzes cosine similarities and next-token prediction changes
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from pathlib import Path
import pandas as pd

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

# Optimize CUDA memory allocation to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def filter_sentences_by_prediction(sentences: List[str], target_word: str, 
                                  model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[str]:
    """
    Filter sentences to only include those where the token before target_word
    predicts target_word as the highest probability next token.
    """
    print(f"Filtering {len(sentences)} sentences where '{target_word}' has highest next-token probability...")
    
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
        if i % 50 == 0:  # More frequent updates and memory cleanup
            print(f"  Processing sentence {i+1}/{len(sentences)}...")
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Find position before target word
        pos_before_target = find_token_before_animals(tokenizer, sentence) if target_word == "animals" else find_token_before_word(tokenizer, sentence, target_word)
        if pos_before_target is None:
            continue
        
        try:
            # Tokenize and run model
            token_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(token_ids)
                logits = outputs.logits[0, pos_before_target, :]  # Next token logits
                
                # Get the most likely token
                top_token_id = torch.argmax(logits).item()
                
                # Check if the most likely token is our target word
                if top_token_id in target_token_ids:
                    filtered_sentences.append(sentence)
            
            # Clear intermediate tensors
            del token_ids, outputs, logits
                    
        except Exception as e:
            # Skip sentences that cause errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    print(f"Filtered to {len(filtered_sentences)} sentences where '{target_word}' has highest probability")
    return filtered_sentences

def find_token_before_word(tokenizer: AutoTokenizer, sentence: str, target_word: str) -> Optional[int]:
    """
    Find the position of the token immediately before the specified target word.
    Returns None if target word is not found or is the first token.
    """
    # Tokenize the sentence
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    target_word_lower = target_word.lower()
    
    # Find target word in the tokenized sequence
    for i, token in enumerate(tokens):
        # For Llama tokenization, remove the leading space indicator ▁
        clean_token = token.replace('▁', '').lower()
        
        # Check if this token matches the target word
        if (clean_token == target_word_lower or 
            clean_token.startswith(target_word_lower) or
            (target_word_lower in clean_token and len(clean_token) <= len(target_word_lower) + 3)):
            
            # Return the position before this token (if it exists and is valid)
            if i > 0:
                return i - 1
            else:
                return None  # target word is the first token
    
    # Target word not found
    return None

def load_animal_sentences(data_file: str, num_sentences: int = 20, 
                         model: Optional[AutoModelForCausalLM] = None, 
                         tokenizer: Optional[AutoTokenizer] = None) -> List[str]:
    """Load diverse animal sentences from the manifold dataset, filtered by prediction accuracy."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if "animals" not in data:
        raise ValueError("'animals' key not found in data file")
    
    all_sentences = data["animals"]
    print(f"Available animal sentences: {len(all_sentences)}")
    
    # Filter sentences if model and tokenizer are provided
    if model is not None and tokenizer is not None:
        filtered_sentences = filter_sentences_by_prediction(all_sentences, "animals", model, tokenizer)
        if len(filtered_sentences) < num_sentences:
            print(f"Warning: Only {len(filtered_sentences)} sentences pass filter, requested {num_sentences}")
            selected_sentences = filtered_sentences
        else:
            # Select diverse sentences from filtered set
            step = len(filtered_sentences) // num_sentences
            selected_sentences = [filtered_sentences[i * step] for i in range(num_sentences)]
    else:
        # Fallback to original selection method
        step = len(all_sentences) // num_sentences
        selected_sentences = [all_sentences[i * step] for i in range(num_sentences)]
    
    print(f"Selected {len(selected_sentences)} diverse animal sentences")
    return selected_sentences

def find_token_before_animals(tokenizer: AutoTokenizer, sentence: str) -> Optional[int]:
    """
    Find the position of the token immediately before the word "animals".
    Returns None if "animals" is not found or is the first token.
    """
    # Tokenize the sentence
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Find "animals" in the tokenized sequence
    for i, token in enumerate(tokens):
        # For Llama tokenization, remove the leading space indicator ▁
        clean_token = token.replace('▁', '').lower()
        
        # Check if this token matches "animals" or starts with it
        if (clean_token == "animals" or 
            clean_token.startswith("animals") or
            ("animal" in clean_token and len(clean_token) <= 8)):
            
            # Return the position before this token (if it exists and is valid)
            if i > 0:
                return i - 1
            else:
                return None  # "animals" is the first token
    
    # "animals" not found
    return None

class FisherInfoExtractor:
    """Extract residual stream activations and compute Fisher Information Matrix."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_activation_before_animals(self, sentence: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Extract residual stream activation h at the token before 'animals'."""
        
        # Find position before "animals"
        pos_before_animals = find_token_before_animals(self.tokenizer, sentence)
        if pos_before_animals is None:
            return None
        
        # Tokenize and prepare input
        token_ids = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(self.device)
        
        # Hook to capture layer output (residual stream)
        activation = None
        
        def hook_fn(module, input, output):
            nonlocal activation
            # Extract hidden states from layer output (output can be tuple or tensor)
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Extract activation at the specified position
            if pos_before_animals < hidden_states.shape[1]:
                activation = hidden_states[0, pos_before_animals, :].detach().clone()  # [hidden_size]
        
        # Register hook on the layer (residual stream output)
        hook = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(token_ids)
            
            return activation
            
        finally:
            hook.remove()
    
    def compute_fisher_matrix(self, h: torch.Tensor, layer_idx: int, 
                            max_vocab_subset: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Fisher Information Matrix I(h) = Σ_c p(c|h)[∇_h log p(c|h) ∇_h log p(c|h)^T]
        
        Args:
            h: Activation vector [hidden_size]
            layer_idx: Layer index where h was extracted
            max_vocab_subset: Limit computation to top-k most likely tokens for efficiency
            
        Returns:
            fisher_matrix: Fisher Information Matrix [hidden_size, hidden_size]
            top_eigenvalue: Largest eigenvalue of the Fisher matrix
        """
        h_requires_grad = h.detach().clone().requires_grad_(True)
        
        # Forward pass from the activation through the remaining layers to get logits
        with torch.enable_grad():
            # We need to reconstruct the forward pass from layer_idx+1 to the output
            # For now, we'll use the language modeling head directly
            # This is an approximation - ideally we'd pass through remaining layers
            
            logits = self.model.lm_head(h_requires_grad)  # [vocab_size]
            probs = F.softmax(logits, dim=-1)  # [vocab_size]
            
            # Limit to top-k tokens for computational efficiency
            top_probs, top_indices = torch.topk(probs, min(max_vocab_subset, probs.shape[0]))
            
            # Initialize Fisher matrix
            fisher_matrix = torch.zeros(h.shape[0], h.shape[0], device=h.device, dtype=h.dtype)
            
            # Compute Fisher matrix components in smaller batches to save memory
            batch_size = 20  # Process 20 tokens at a time to reduce memory usage
            num_batches = (len(top_probs) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(top_probs))
                
                batch_probs = top_probs[start_idx:end_idx]
                batch_indices = top_indices[start_idx:end_idx]
                
                for i, (prob, token_idx) in enumerate(zip(batch_probs, batch_indices)):
                    if prob < 1e-6:  # Skip very low probability tokens (increased threshold)
                        continue
                    
                    # Compute gradient of log p(c|h) w.r.t. h
                    log_prob = torch.log(prob + 1e-8)  # Add small epsilon for numerical stability
                    
                    # Clear gradients
                    if h_requires_grad.grad is not None:
                        h_requires_grad.grad.zero_()
                    
                    # Compute gradient
                    log_prob.backward(retain_graph=True)
                    grad = h_requires_grad.grad.clone()
                    
                    # Add to Fisher matrix: p(c|h) * grad * grad^T
                    fisher_matrix += prob * torch.outer(grad, grad)
                    
                    # Clear intermediate computations
                    del grad
                
                # Clear CUDA cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Eigendecomposition to find top eigenvector
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(fisher_matrix)
            # Sort in descending order
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            top_eigenvalue = eigenvalues[sorted_indices[0]]
            top_eigenvector = eigenvectors[:, sorted_indices[0]]
            
            return top_eigenvector, top_eigenvalue
            
        except Exception as e:
            print(f"Warning: Eigendecomposition failed: {e}")
            # Return random direction as fallback
            random_direction = torch.randn_like(h)
            random_direction = random_direction / torch.norm(random_direction)
            return random_direction, torch.tensor(1.0, device=h.device)

class FisherPerturbationAnalyzer:
    """Analyze the effects of Fisher-guided perturbations."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def perturb_and_analyze(self, sentence: str, layer_idx: int, 
                          h_original: torch.Tensor, fisher_direction: torch.Tensor, 
                          eigenvalue: torch.Tensor, scales: List[float]) -> Dict:
        """
        Perturb h along Fisher direction and analyze next-token predictions.
        
        Args:
            sentence: Original sentence
            layer_idx: Layer where perturbation is applied
            h_original: Original activation
            fisher_direction: Top Fisher eigenvector
            eigenvalue: Corresponding eigenvalue
            scales: List of perturbation scales α
            
        Returns:
            Dictionary containing perturbation results
        """
        results = {}
        
        # Find position before "animals" for injection
        pos_before_animals = find_token_before_animals(self.tokenizer, sentence)
        if pos_before_animals is None:
            return {"error": "Could not find position before 'animals'"}
        
        token_ids = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors="pt").to(self.device)
        
        for scale in scales:
            # Compute perturbation
            perturbation = scale * eigenvalue * fisher_direction
            h_perturbed = h_original + perturbation
            
            # Hook to inject perturbed activation
            def make_injection_hook(h_new):
                def injection_hook(module, input, output):
                    # Handle both tuple and tensor outputs
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        if pos_before_animals < hidden_states.shape[1]:
                            hidden_states[0, pos_before_animals, :] = h_new
                        return output
                    else:
                        if pos_before_animals < output.shape[1]:
                            output[0, pos_before_animals, :] = h_new
                        return output
                return injection_hook
            
            # Apply perturbation and get next-token predictions
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                make_injection_hook(h_perturbed)
            )
            
            try:
                with torch.no_grad():
                    outputs = self.model(token_ids)
                    logits = outputs.logits[0, pos_before_animals, :]  # Next token logits
                    probs = F.softmax(logits, dim=-1)
                    
                    # Get top-5 predictions
                    top_probs, top_indices = torch.topk(probs, 5)
                    top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                    
                    results[scale] = {
                        "top_tokens": top_tokens,
                        "top_probs": top_probs.cpu().tolist()
                    }
                    
            finally:
                hook.remove()
        
        return results

def compute_cosine_similarities(fisher_directions: Dict[str, Dict[int, torch.Tensor]], 
                               layers: List[int]) -> np.ndarray:
    """Compute cosine similarities among Fisher eigenvectors."""
    
    # Collect all Fisher directions
    directions = []
    labels = []
    
    for sentence_idx in sorted(fisher_directions.keys()):
        for layer_idx in layers:
            if layer_idx in fisher_directions[sentence_idx]:
                direction = fisher_directions[sentence_idx][layer_idx]
                directions.append(direction.cpu().numpy())
                labels.append(f"S{sentence_idx}_L{layer_idx}")
    
    if not directions:
        return np.array([]), []
    
    # Compute cosine similarity matrix
    directions = np.array(directions)
    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions_normalized = directions / (norms + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(directions_normalized, directions_normalized.T)
    
    return similarity_matrix, labels

def plot_similarity_matrix(similarity_matrix: np.ndarray, labels: List[str], output_dir: str):
    """Plot the cosine similarity matrix of Fisher eigenvectors."""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=False,
                cmap='coolwarm',
                center=0,
                square=True)
    plt.title('Cosine Similarity Matrix: Fisher Eigenvectors')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fisher_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_prediction_changes(all_results: Dict, scales: List[float], output_dir: str):
    """Analyze and visualize changes in top-5 predictions across scales."""
    
    # Aggregate results across sentences
    aggregated_changes = {}
    
    for scale in scales:
        token_counts = {}
        for sentence_results in all_results.values():
            for layer_results in sentence_results.values():
                if scale in layer_results and "top_tokens" in layer_results[scale]:
                    for token in layer_results[scale]["top_tokens"]:
                        token_counts[token] = token_counts.get(token, 0) + 1
        
        # Get most frequent tokens for this scale
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        aggregated_changes[scale] = sorted_tokens[:10]  # Top 10 most frequent
    
    # Plot the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, scale in enumerate(scales[:6]):  # Plot first 6 scales
        if i < len(axes):
            tokens, counts = zip(*aggregated_changes[scale]) if aggregated_changes[scale] else ([], [])
            axes[i].bar(range(len(tokens)), counts)
            axes[i].set_title(f'Scale α = {scale}')
            axes[i].set_xlabel('Top Predicted Tokens')
            axes[i].set_ylabel('Frequency')
            if tokens:
                axes[i].set_xticks(range(len(tokens)))
                axes[i].set_xticklabels(tokens, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/perturbation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def clear_memory():
    """Aggressive memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_fisher_experiment(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    data_file: str = "../tools/manifold_sentences_hard_exactword_1000.json",
    num_sentences: int = 20,
    layers: List[int] = [0, 5, 10, 15, 20, 25, 30, 31],
    scales: List[float] = [0, -10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10],
    output_dir: str = "fisher_results",
    max_vocab_subset: int = 200,
    batch_size: int = 20
):
    """Run the complete Fisher-guided perturbation experiment."""
    
    print("=" * 60)
    print("FISHER-GUIDED EMBEDDING PERTURBATIONS EXPERIMENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Memory optimizations: vocab_subset={max_vocab_subset}, batch_size={batch_size}")
    
    # Clear any existing memory
    clear_memory()
    
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
            print("   Loading model for CPU")
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
        print(f"❌ Error loading model: {e}")
        raise
    
    # 2. Load sentences (with filtering based on prediction accuracy)
    print("\n2. Loading and filtering animal sentences...")
    sentences = load_animal_sentences(data_file, num_sentences, model, tokenizer)
    
    # 3. Initialize extractors and analyzers
    print("\n3. Initializing Fisher analysis components...")
    fisher_extractor = FisherInfoExtractor(model, tokenizer)
    perturbation_analyzer = FisherPerturbationAnalyzer(model, tokenizer)
    
    # 4. Main experimental loop
    print("\n4. Running Fisher analysis across sentences and layers...")
    Path(output_dir).mkdir(exist_ok=True)
    
    fisher_directions = {}  # sentence_idx -> layer_idx -> direction
    all_results = {}  # sentence_idx -> layer_idx -> scale -> results
    
    for sentence_idx, sentence in enumerate(sentences):
        print(f"\nSentence {sentence_idx + 1}/{len(sentences)}:")
        print(f"  '{sentence[:80]}...'")
        
        fisher_directions[sentence_idx] = {}
        all_results[sentence_idx] = {}
        
        for layer_idx in layers:
            print(f"  Layer {layer_idx}:", end=" ")
            
            # Extract activation before "animals"
            h = fisher_extractor.extract_activation_before_animals(sentence, layer_idx)
            if h is None:
                print("❌ Could not extract activation")
                continue
            
            # Compute Fisher matrix and top eigenvector
            try:
                fisher_direction, eigenvalue = fisher_extractor.compute_fisher_matrix(h, layer_idx, max_vocab_subset)
                fisher_directions[sentence_idx][layer_idx] = fisher_direction
                print(f"✅ Fisher eigenvalue: {eigenvalue.item():.6f}")
                
                # Perturbation analysis
                results = perturbation_analyzer.perturb_and_analyze(
                    sentence, layer_idx, h, fisher_direction, eigenvalue, scales
                )
                all_results[sentence_idx][layer_idx] = results
                
                # Clear memory after each computation
                del h, fisher_direction, eigenvalue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Clear memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    # 5. Compute and plot similarity matrix
    print("\n5. Computing cosine similarities...")
    similarity_matrix, labels = compute_cosine_similarities(fisher_directions, layers)
    if similarity_matrix.size > 0:
        plot_similarity_matrix(similarity_matrix, labels, output_dir)
        print(f"   Saved similarity matrix to {output_dir}/fisher_similarity_matrix.png")
    
    # 6. Analyze prediction changes
    print("\n6. Analyzing prediction changes...")
    analyze_prediction_changes(all_results, scales, output_dir)
    print(f"   Saved prediction analysis to {output_dir}/perturbation_analysis.png")
    
    # 7. Save results
    print("\n7. Saving results...")
    results = {
        'experiment_config': {
            'model_name': model_name,
            'num_sentences': num_sentences,
            'layers': layers,
            'scales': scales
        },
        'sentences': sentences,
        'perturbation_results': all_results,
        'similarity_matrix': similarity_matrix.tolist() if similarity_matrix.size > 0 else [],
        'similarity_labels': labels
    }
    
    # Convert tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    results = convert_tensors(results)
    
    with open(f"{output_dir}/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   Results saved to {output_dir}/experiment_results.json")
    print("\n✅ Experiment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fisher-guided perturbation experiment")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B",
                       help="Model name")
    parser.add_argument("--data-file", default="../tools/manifold_sentences_hard_exactword_1000.json",
                       help="Path to sentences data file")
    parser.add_argument("--num-sentences", type=int, default=20,
                       help="Number of animal sentences to analyze")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30, 31],
                       help="Layer indices to analyze")
    parser.add_argument("--output-dir", default="fisher_results",
                       help="Output directory")
    parser.add_argument("--max-vocab-subset", type=int, default=200,
                       help="Maximum number of top vocabulary tokens to use for Fisher computation (reduce for memory savings)")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for Fisher computation (reduce for memory savings)")
    
    args = parser.parse_args()
    
    run_fisher_experiment(
        model_name=args.model_name,
        data_file=args.data_file,
        num_sentences=args.num_sentences,
        layers=args.layers,
        output_dir=args.output_dir,
        max_vocab_subset=args.max_vocab_subset,
        batch_size=args.batch_size
    ) 