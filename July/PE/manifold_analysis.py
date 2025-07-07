"""
Robust manifold analysis module for concept representation.
Fixes centering issues and validates semantic structure of extracted manifolds.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

from config import (
    ExperimentConfig, SYSTEM_PROMPT, MANIFOLD_VALIDATION_CRITERIA,
    DEVICE, TORCH_DTYPE
)

@dataclass
class ManifoldAnalysis:
    """Results of concept manifold analysis"""
    concept: str
    layer: int
    pca: PCA
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    explained_variance_ratio: torch.Tensor
    centered_activations: torch.Tensor
    concept_centroid: torch.Tensor
    effective_pcs: List[int]
    semantic_correlations: Dict[str, float]
    cross_validation_stability: float

class ConceptManifoldAnalyzer:
    """Analyzes concept manifolds with proper validation and semantic interpretation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the language model and tokenizer"""
        print(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded successfully")
    
    def extract_final_token_activations(
        self, 
        prompts: List[str], 
        layer_idx: int,
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Extract final token activations from the specified layer.
        
        Args:
            prompts: List of prompts to process
            layer_idx: Layer index to extract activations from
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape [num_prompts, hidden_dim] with final token activations
        """
        if self.model is None:
            self.load_model()
        
        activations = []
        
        def hook_fn(module, input, output):
            # Extract final token activation for each sequence in the batch
            final_token_acts = output[0][:, -1, :].detach().cpu()
            activations.append(final_token_acts)
        
        # Register hook
        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            # Process prompts in batches
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Extracting layer {layer_idx} activations"):
                batch_prompts = prompts[i:i+batch_size]
                
                # Prepare messages for each prompt in batch
                batch_messages = []
                for prompt in batch_prompts:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    batch_messages.append(messages)
                
                # Tokenize batch
                batch_inputs = []
                for messages in batch_messages:
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        return_tensors="pt",
                        add_generation_prompt=False,
                        padding=False,  # We'll handle padding manually
                        truncation=True,
                        max_length=512
                    )
                    batch_inputs.append(inputs.squeeze(0))
                
                # Pad to same length
                max_len = max(inp.shape[0] for inp in batch_inputs)
                padded_inputs = []
                attention_masks = []
                
                for inp in batch_inputs:
                    pad_len = max_len - inp.shape[0]
                    if pad_len > 0:
                        padded_inp = torch.cat([
                            torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=inp.dtype),
                            inp
                        ])
                        attention_mask = torch.cat([
                            torch.zeros(pad_len, dtype=torch.long),
                            torch.ones(inp.shape[0], dtype=torch.long)
                        ])
                    else:
                        padded_inp = inp
                        attention_mask = torch.ones(inp.shape[0], dtype=torch.long)
                    
                    padded_inputs.append(padded_inp)
                    attention_masks.append(attention_mask)
                
                # Stack into batch tensors
                batch_input_ids = torch.stack(padded_inputs).to(self.config.device)
                batch_attention_mask = torch.stack(attention_masks).to(self.config.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask
                    )
                
                # Clear GPU memory
                del batch_input_ids, batch_attention_mask, outputs
                torch.cuda.empty_cache()
                gc.collect()
        
        finally:
            hook_handle.remove()
        
        # Concatenate all activations
        if activations:
            return torch.cat(activations, dim=0)
        else:
            return torch.empty(0, self.model.config.hidden_size)
    
    def analyze_concept_manifold(
        self,
        concept: str,
        concept_prompts: Dict[str, List[str]], 
        layer_idx: int,
        validate_stability: bool = True
    ) -> ManifoldAnalysis:
        """
        Analyze the manifold for a specific concept at a given layer.
        
        Args:
            concept: The concept name
            concept_prompts: Dictionary mapping semantic categories to prompts
            layer_idx: Layer to analyze
            validate_stability: Whether to perform cross-validation for stability
            
        Returns:
            ManifoldAnalysis object with complete analysis results
        """
        print(f"\nAnalyzing {concept} manifold at layer {layer_idx}")
        
        # Flatten all prompts while keeping track of categories
        all_prompts = []
        prompt_categories = []
        
        for category, prompts in concept_prompts.items():
            all_prompts.extend(prompts)
            prompt_categories.extend([category] * len(prompts))
        
        print(f"Processing {len(all_prompts)} prompts across {len(concept_prompts)} semantic categories")
        
        # Extract activations
        activations = self.extract_final_token_activations(all_prompts, layer_idx)
        
        if activations.shape[0] == 0:
            raise ValueError(f"No activations extracted for concept {concept}")
        
        print(f"Extracted activations shape: {activations.shape}")
        
        # Move to CPU for PCA analysis
        activations_cpu = activations.cpu().float()
        
        # Proper centering: subtract concept-specific mean (not global mean)
        concept_centroid = activations_cpu.mean(dim=0)
        centered_activations = activations_cpu - concept_centroid
        
        print(f"Centered activations, centroid norm: {concept_centroid.norm():.3f}")
        
        # Perform PCA
        pca = PCA()
        pca.fit(centered_activations.numpy())
        
        # Convert results to tensors
        eigenvectors = torch.tensor(pca.components_, dtype=torch.float32)
        eigenvalues = torch.tensor(pca.explained_variance_, dtype=torch.float32)
        explained_variance_ratio = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32)
        
        # Identify effective PCs based on explained variance threshold
        variance_threshold = MANIFOLD_VALIDATION_CRITERIA['min_explained_variance']
        effective_pcs = [i for i, ratio in enumerate(explained_variance_ratio) 
                        if ratio > variance_threshold]
        effective_pcs = effective_pcs[:MANIFOLD_VALIDATION_CRITERIA['max_pcs_to_analyze']]
        
        print(f"Found {len(effective_pcs)} effective PCs explaining >{variance_threshold:.1%} variance each")
        print(f"Top 5 PC explained variance: {explained_variance_ratio[:5].tolist()}")
        
        # Analyze semantic correlations
        semantic_correlations = self._analyze_semantic_correlations(
            centered_activations, eigenvectors, prompt_categories, effective_pcs
        )
        
        # Cross-validation stability (if requested)
        stability_score = 0.0
        if validate_stability and len(all_prompts) >= 3 * MANIFOLD_VALIDATION_CRITERIA['cross_validation_folds']:
            stability_score = self._cross_validate_manifold_stability(
                centered_activations, MANIFOLD_VALIDATION_CRITERIA['cross_validation_folds']
            )
            print(f"Cross-validation stability score: {stability_score:.3f}")
        
        return ManifoldAnalysis(
            concept=concept,
            layer=layer_idx,
            pca=pca,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained_variance_ratio,
            centered_activations=centered_activations,
            concept_centroid=concept_centroid,
            effective_pcs=effective_pcs,
            semantic_correlations=semantic_correlations,
            cross_validation_stability=stability_score
        )
    
    def _analyze_semantic_correlations(
        self,
        centered_activations: torch.Tensor,
        eigenvectors: torch.Tensor,
        prompt_categories: List[str],
        effective_pcs: List[int]
    ) -> Dict[str, float]:
        """
        Analyze correlations between principal components and semantic categories.
        
        Args:
            centered_activations: Centered activation matrix
            eigenvectors: Principal component vectors
            prompt_categories: Category label for each prompt
            effective_pcs: List of effective PC indices
            
        Returns:
            Dictionary mapping PC descriptions to correlation scores
        """
        correlations = {}
        
        # Project activations onto each effective PC
        for pc_idx in effective_pcs:
            pc_vector = eigenvectors[pc_idx]
            projections = centered_activations @ pc_vector
            
            # Analyze correlation with semantic categories
            unique_categories = list(set(prompt_categories))
            category_means = {}
            
            for category in unique_categories:
                category_indices = [i for i, cat in enumerate(prompt_categories) if cat == category]
                if category_indices:
                    category_projections = projections[category_indices]
                    category_means[category] = category_projections.mean().item()
            
            # Calculate the range of category means as a measure of semantic separation
            if len(category_means) > 1:
                mean_values = list(category_means.values())
                semantic_separation = max(mean_values) - min(mean_values)
                correlations[f'PC{pc_idx}_semantic_separation'] = semantic_separation
                
                # Store category means for interpretation
                correlations[f'PC{pc_idx}_category_means'] = category_means
        
        return correlations
    
    def _cross_validate_manifold_stability(
        self,
        centered_activations: torch.Tensor,
        n_folds: int = 3
    ) -> float:
        """
        Perform cross-validation to assess manifold stability.
        
        Args:
            centered_activations: Centered activation matrix
            n_folds: Number of cross-validation folds
            
        Returns:
            Average cosine similarity between PC1 vectors across folds
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_pc1s = []
        
        activations_np = centered_activations.numpy()
        
        for train_idx, _ in kf.split(activations_np):
            # Fit PCA on training fold
            train_activations = activations_np[train_idx]
            fold_pca = PCA()
            fold_pca.fit(train_activations)
            
            # Store the first principal component
            fold_pc1s.append(fold_pca.components_[0])
        
        # Calculate pairwise cosine similarities between PC1 vectors
        similarities = []
        for i in range(len(fold_pc1s)):
            for j in range(i + 1, len(fold_pc1s)):
                sim = cosine_similarity([fold_pc1s[i]], [fold_pc1s[j]])[0, 0]
                similarities.append(abs(sim))  # Use absolute value since direction can flip
        
        return np.mean(similarities)
    
    def analyze_all_concepts(
        self,
        dataset: Dict[str, Dict[str, List[str]]],
        target_layers: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, ManifoldAnalysis]]:
        """
        Analyze manifolds for all concepts across specified layers.
        
        Args:
            dataset: Dataset mapping concepts to semantic categories to prompts
            target_layers: List of layers to analyze (defaults to config)
            
        Returns:
            Nested dictionary: {concept: {layer: ManifoldAnalysis}}
        """
        if target_layers is None:
            target_layers = self.config.target_layers
        
        results = {}
        
        for concept in dataset.keys():
            print(f"\n{'='*60}")
            print(f"Analyzing concept: {concept}")
            print(f"{'='*60}")
            
            results[concept] = {}
            
            for layer in target_layers:
                try:
                    analysis = self.analyze_concept_manifold(
                        concept, dataset[concept], layer, validate_stability=True
                    )
                    results[concept][layer] = analysis
                    
                    # Print summary
                    print(f"\nLayer {layer} summary:")
                    print(f"  - Effective PCs: {len(analysis.effective_pcs)}")
                    print(f"  - Top PC variance: {analysis.explained_variance_ratio[0]:.3f}")
                    print(f"  - Stability score: {analysis.cross_validation_stability:.3f}")
                    
                except Exception as e:
                    print(f"Error analyzing {concept} at layer {layer}: {e}")
                    continue
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
        
        return results
    
    def save_analysis_results(
        self,
        results: Dict[str, Dict[int, ManifoldAnalysis]],
        filepath: str
    ):
        """Save analysis results to disk (serializable parts only)"""
        def convert_to_serializable(obj):
            """Convert numpy/torch types to JSON serializable types"""
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return obj.item()
            elif hasattr(obj, 'item') and callable(obj.item):
                # Handle any numpy scalar types
                try:
                    return obj.item()
                except:
                    return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                # Handle any numpy array-like objects
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                # Last resort: try to convert to float if possible
                try:
                    return float(obj)
                except (TypeError, ValueError):
                    return str(obj)
        
        serializable_results = {}
        
        for concept, layer_results in results.items():
            serializable_results[concept] = {}
            
            for layer, analysis in layer_results.items():
                serializable_results[concept][layer] = {
                    'concept': analysis.concept,
                    'layer': analysis.layer,
                    'eigenvalues': convert_to_serializable(analysis.eigenvalues),
                    'explained_variance_ratio': convert_to_serializable(analysis.explained_variance_ratio),
                    'effective_pcs': convert_to_serializable(analysis.effective_pcs),
                    'semantic_correlations': convert_to_serializable(analysis.semantic_correlations),
                    'cross_validation_stability': convert_to_serializable(analysis.cross_validation_stability),
                    'centroid_norm': convert_to_serializable(analysis.concept_centroid.norm())
                }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis results saved to {filepath}")

def validate_manifold_quality(analysis: ManifoldAnalysis) -> Dict[str, bool]:
    """
    Validate the quality of an extracted manifold.
    
    Args:
        analysis: ManifoldAnalysis object to validate
        
    Returns:
        Dictionary of validation checks and their results
    """
    validation = {}
    
    # Check if we have sufficient effective PCs
    validation['sufficient_pcs'] = len(analysis.effective_pcs) >= 2
    
    # Check if top PC explains reasonable variance
    validation['meaningful_variance'] = analysis.explained_variance_ratio[0] > 0.1
    
    # Check stability (if available)
    if analysis.cross_validation_stability > 0:
        validation['stable_manifold'] = analysis.cross_validation_stability > 0.3
    else:
        validation['stable_manifold'] = True  # Not tested
    
    # Check for semantic separation
    has_semantic_separation = any(
        'semantic_separation' in key and value > 0.1 
        for key, value in analysis.semantic_correlations.items()
        if isinstance(value, (int, float))
    )
    validation['semantic_structure'] = has_semantic_separation
    
    return validation

if __name__ == "__main__":
    # Example usage
    from data_preparation import ConceptDatasetGenerator
    
    config = ExperimentConfig()
    analyzer = ConceptManifoldAnalyzer(config)
    
    # Generate test dataset
    generator = ConceptDatasetGenerator()
    dataset = generator.create_balanced_dataset(['dog'])
    
    # Analyze manifolds
    results = analyzer.analyze_all_concepts(dataset, target_layers=[8, 16])
    
    # Validate results
    for concept, layer_results in results.items():
        for layer, analysis in layer_results.items():
            validation = validate_manifold_quality(analysis)
            print(f"{concept} layer {layer} validation: {validation}")
    
    # Save results
    analyzer.save_analysis_results(results, "manifold_analysis_results.json") 