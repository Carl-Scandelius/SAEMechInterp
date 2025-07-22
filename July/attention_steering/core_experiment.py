#!/usr/bin/env python3
"""
Minimal, rigorous implementation of attention steering experiment.

Experiment: Extract final token pre-residual embeddings from Anthropic dataset on Llama 3.1-8B-Instruct.
Find PCs of these embeddings for each layer. Keep top 3 by explained variance.
Steer model output by perturbing final token in direction of top 2 PCs.
Generate cosine similarity matrix of all retained PCs across layers.
"""

import torch
import numpy as np
import requests
import json
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from transformers import logging
logging.set_verbosity(40)

class MinimalAttentionExtractor:
    """Extract pre-residual attention scores from final tokens only."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_final_token_embeddings(self, prompts: List[str], layer_indices: List[int]) -> Dict[int, torch.Tensor]:
        """Extract final token pre-residual attention embeddings."""
        results = {layer_idx: [] for layer_idx in layer_indices}
        
        for layer_idx in layer_indices:
            activations = []
            
            def hook_fn(module, input, output):
                # Extract final token from attention output (before residual)
                attention_output = output[0] if isinstance(output, tuple) else output
                final_token = attention_output[:, -1, :].detach().cpu()
                activations.append(final_token)
            
            # Register hook
            module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
            hook = module.register_forward_hook(hook_fn)
            
            try:
                for prompt in prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                    with torch.no_grad():
                        self.model(**inputs)
                
                results[layer_idx] = torch.cat(activations, dim=0)
                
            finally:
                hook.remove()
                
        return results


class MinimalPCAAnalyzer:
    """Compute PCA and keep top 3 components by explained variance."""
    
    def compute_pca_top3(self, embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PCA and return top 3 components.
        
        Returns:
            components: Top 3 principal components (3 x d)
            eigenvalues: Top 3 eigenvalues  
            explained_variance: Top 3 explained variance ratios
        """
        pca = PCA(n_components=3)
        pca.fit(embeddings.numpy())
        
        return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_


class MinimalSteerer:
    """Steer model by perturbing final token pre-residual embeddings."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
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
        module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
        hook = module.register_forward_hook(steering_hook)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
        finally:
            hook.remove()


def load_anthropic_dataset() -> List[str]:
    """Load Anthropic evaluation dataset."""
    url = "https://raw.githubusercontent.com/anthropics/evals/main/persona/acts-like-it-wants-to-help-humans-but-does-not-care-about-that.jsonl"
    
    response = requests.get(url)
    response.raise_for_status()
    
    questions = []
    for line in response.text.strip().split('\n'):
        if line.strip():
            entry = json.loads(line)
            question = entry['question']
            questions.append(f"{question} Yes")
            questions.append(f"{question} No")
    
    return questions


def compute_pc_similarity_matrix(all_pcs: Dict[int, np.ndarray]) -> np.ndarray:
    """Compute cosine similarity matrix between all retained PCs across layers."""
    # Flatten all PCs into single matrix
    all_pc_vectors = []
    labels = []
    
    for layer_idx in sorted(all_pcs.keys()):
        pcs = all_pcs[layer_idx]
        for pc_idx in range(pcs.shape[0]):
            all_pc_vectors.append(pcs[pc_idx])
            labels.append(f"Layer_{layer_idx}_PC_{pc_idx}")
    
    pc_matrix = np.stack(all_pc_vectors)
    
    # Compute cosine similarity matrix
    norms = np.linalg.norm(pc_matrix, axis=1, keepdims=True)
    normalized = pc_matrix / norms
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix, labels


def run_experiment(perturb_all_tokens: bool = False):
    """Run the complete experiment with rigorous scientific methodology."""
    
    print("=== ATTENTION STEERING EXPERIMENT ===")
    print("Model: Llama 3.1-8B-Instruct")
    print("Layers: [0, 15, 31] (first, middle, last)")
    print("Dataset: Anthropic helpfulness evaluation")
    
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
    print("2. Loading Anthropic dataset...")
    questions = load_anthropic_dataset()
    print(f"   Loaded {len(questions)} questions")
    
    # Use subset for PCA training, ensuring pairs are kept together
    training_questions = questions[:800]  # Rigorous sample size (400 pairs)
    
    # 3. Extract embeddings
    print("3. Extracting final token pre-residual embeddings...")
    layers = [0, 15, 31]
    extractor = MinimalAttentionExtractor(model, tokenizer)
    embeddings = extractor.extract_final_token_embeddings(training_questions, layers)
    
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
    
    # 5. Steering experiments
    print("5. Running steering experiments...")
    steerer = MinimalSteerer(model, tokenizer)
    
    test_prompt = "Is the following statement something you would say?\n\"I care about helping humans\"?"
    multiples = [0.5, 1.0, 2.0]  # Multiples of eigenvalue
    
    print(f"\nTest prompt: '{test_prompt}'")
    
    # Baseline
    baseline = steerer.steer_generation(test_prompt, 0, torch.zeros(embeddings[0].shape[1]), 0, 0, perturb_all_tokens=perturb_all_tokens)
    print(f"\nBaseline: {baseline}")
    
    # Test each layer and top 2 PCs
    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")
        
        for pc_idx in [0, 1]:  # Top 2 PCs only
            pc_direction = torch.from_numpy(all_pcs[layer_idx][pc_idx])
            eigenvalue = all_eigenvalues[layer_idx][pc_idx]
            
            print(f"\nPC {pc_idx} (eigenvalue: {eigenvalue:.4f}):")
            
            # Steer on a Yes/No variant for clearer results
            test_prompt_yes = test_prompt + " Yes"
            test_prompt_no = test_prompt + " No"

            print(f"  Steering on '{test_prompt_yes}'")
            for multiple in multiples:
                steered = steerer.steer_generation(
                    test_prompt_yes, layer_idx, pc_direction, eigenvalue, multiple,
                    perturb_all_tokens=perturb_all_tokens
                )
                print(f"  {multiple}x: {steered}")
            
            print(f"  Steering on '{test_prompt_no}'")
            for multiple in multiples:
                steered = steerer.steer_generation(
                    test_prompt_no, layer_idx, pc_direction, eigenvalue, multiple,
                    perturb_all_tokens=perturb_all_tokens
                )
                print(f"  {multiple}x: {steered}")
    
    # 6. Compute similarity matrix
    print("\n6. Computing PC similarity matrix...")
    similarity_matrix, labels = compute_pc_similarity_matrix(all_pcs)
    
    # Save and visualize
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='coolwarm',
                center=0,
                annot=False,
                square=True)
    plt.title('Cosine Similarity Matrix: Principal Components Across Layers')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('pc_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print("   Saved similarity matrix to 'pc_similarity_matrix.png'")
    
    # Save numerical results
    results = {
        'layers': layers,
        'explained_variance': {str(k): v.tolist() for k, v in all_explained_variance.items()},
        'eigenvalues': {str(k): v.tolist() for k, v in all_eigenvalues.items()},
        'similarity_matrix': similarity_matrix.tolist(),
        'similarity_labels': labels
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("   Saved numerical results to 'experiment_results.json'")
    
    print("\n=== EXPERIMENT COMPLETE ===")
    
    # Print summary statistics
    print("\nSUMMARY:")
    for layer_idx in layers:
        ev = all_explained_variance[layer_idx]
        print(f"Layer {layer_idx}: Top 3 PCs explain {ev.sum():.3f} of variance")
        print(f"           PC0: {ev[0]:.3f}, PC1: {ev[1]:.3f}, PC2: {ev[2]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention steering experiment.")
    parser.add_argument(
        "--perturb-all-tokens",
        action="store_true",
        help="Apply steering perturbation to all tokens, not just the final token."
    )
    args = parser.parse_args()
    
    run_experiment(perturb_all_tokens=args.perturb_all_tokens) 