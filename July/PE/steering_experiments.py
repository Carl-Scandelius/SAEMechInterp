"""
Main steering experiments module implementing multiple perturbation strategies.
Fixes issues in original perturbation code with proper activation handling.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Callable
import json
import gc
from tqdm import tqdm
from dataclasses import dataclass

from config import ExperimentConfig, SYSTEM_PROMPT, PROMPT_TEMPLATES
from manifold_analysis import ManifoldAnalysis
from evaluation_metrics import EvaluationResult, SteeringEffectivenessEvaluator

@dataclass
class SteeringConfig:
    """Configuration for a steering experiment"""
    concept: str
    layer: int
    pc_axis: int
    target_dimension: str
    perturbation_strategy: str
    perturbation_scale: float
    evaluation_prompts: List[str]

class ActivationSteerer:
    """Handles activation steering with multiple intervention strategies"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.active_hooks = []
        
    def load_model(self):
        """Load the language model and tokenizer"""
        if self.model is None:
            print(f"Loading model: {self.config.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Model loaded successfully")
    
    def extract_final_token_activation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Extract the final token activation for a single prompt.
        
        Args:
            prompt: Input prompt
            layer_idx: Layer to extract from
            
        Returns:
            Final token activation tensor
        """
        if self.model is None:
            self.load_model()
        
        activation = None
        
        def hook_fn(module, input, output):
            nonlocal activation
            # Extract final token activation
            activation = output[0][:, -1, :].detach().clone()
        
        # Register hook
        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,  # Important: prepare for generation
                padding=False,
                truncation=True,
                max_length=512
            )
            inputs = inputs.to(self.config.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=inputs)
            
        finally:
            hook_handle.remove()
        
        if activation is None:
            raise ValueError("Failed to extract activation")
        
        return activation.squeeze(0)  # Remove batch dimension
    
    def create_steering_hook(
        self,
        perturbation_vector: torch.Tensor,
        perturbation_scale: float,
        strategy: str = 'additive'
    ) -> Callable:
        """
        Create a hook function for steering activations.
        
        Args:
            perturbation_vector: Vector to use for perturbation
            perturbation_scale: Scale factor for perturbation
            strategy: Perturbation strategy ('additive', 'projection_based', 'replacement')
            
        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            # Modify the final token activation only
            if strategy == 'additive':
                # Simple additive perturbation
                output[0][:, -1, :] += perturbation_scale * perturbation_vector
                
            elif strategy == 'projection_based':
                # Project current activation onto perturbation direction and modify
                current_activation = output[0][:, -1, :]
                projection = torch.dot(current_activation.squeeze(), perturbation_vector)
                normalized_vector = perturbation_vector / (perturbation_vector.norm() + 1e-8)
                output[0][:, -1, :] += perturbation_scale * projection * normalized_vector
                
            elif strategy == 'replacement':
                # Replace component in perturbation direction
                current_activation = output[0][:, -1, :]
                normalized_vector = perturbation_vector / (perturbation_vector.norm() + 1e-8)
                
                # Remove current component in this direction
                current_component = torch.dot(current_activation.squeeze(), normalized_vector)
                output[0][:, -1, :] -= current_component * normalized_vector
                
                # Add desired component
                output[0][:, -1, :] += perturbation_scale * normalized_vector
                
            else:
                raise ValueError(f"Unknown perturbation strategy: {strategy}")
        
        return hook_fn
    
    def generate_with_steering(
        self,
        prompt: str,
        manifold_analysis: ManifoldAnalysis,
        pc_axis: int,
        perturbation_scale: float,
        strategy: str = 'additive'
    ) -> str:
        """
        Generate text with activation steering.
        
        Args:
            prompt: Input prompt
            manifold_analysis: ManifoldAnalysis containing perturbation vectors
            pc_axis: Principal component axis to use for steering
            perturbation_scale: Scale factor for perturbation
            strategy: Perturbation strategy
            
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()
        
        # Get perturbation vector (eigenvector for the specified PC)
        if pc_axis >= len(manifold_analysis.eigenvectors):
            raise ValueError(f"PC axis {pc_axis} not available (only {len(manifold_analysis.eigenvectors)} PCs)")
        
        perturbation_vector = manifold_analysis.eigenvectors[pc_axis].to(self.config.device)
        
        # Create and register steering hook
        hook_fn = self.create_steering_hook(perturbation_vector, perturbation_scale, strategy)
        hook_handle = self.model.model.layers[manifold_analysis.layer].register_forward_hook(hook_fn)
        
        try:
            # Prepare messages for generation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                padding=False,
                truncation=True,
                max_length=512
            )
            inputs = inputs.to(self.config.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            new_tokens = outputs[:, inputs.shape[-1]:]
            generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            
        finally:
            hook_handle.remove()
        
        return generated_text.strip()
    
    def generate_baseline(self, prompt: str) -> str:
        """Generate text without any steering for comparison"""
        if self.model is None:
            self.load_model()
        
        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            padding=False,
            truncation=True,
            max_length=512
        )
        inputs = inputs.to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[:, inputs.shape[-1]:]
        generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        
        return generated_text.strip()

class SteeringExperimentRunner:
    """Runs comprehensive steering experiments with proper evaluation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.steerer = ActivationSteerer(config)
        self.evaluator = SteeringEffectivenessEvaluator()
        
    def run_single_experiment(
        self,
        steering_config: SteeringConfig,
        manifold_analysis: ManifoldAnalysis
    ) -> List[EvaluationResult]:
        """
        Run a single steering experiment configuration.
        
        Args:
            steering_config: Configuration for this experiment
            manifold_analysis: Manifold analysis results to use for steering
            
        Returns:
            List of evaluation results for each test prompt
        """
        print(f"\nRunning experiment: {steering_config.concept} "
              f"layer {steering_config.layer} PC{steering_config.pc_axis} "
              f"{steering_config.target_dimension} scale {steering_config.perturbation_scale}")
        
        results = []
        
        for prompt_idx, prompt in enumerate(steering_config.evaluation_prompts):
            print(f"  Processing prompt {prompt_idx + 1}/{len(steering_config.evaluation_prompts)}")
            
            try:
                # Generate baseline
                baseline_text = self.steerer.generate_baseline(prompt)
                
                # Generate with steering
                steered_text = self.steerer.generate_with_steering(
                    prompt,
                    manifold_analysis,
                    steering_config.pc_axis,
                    steering_config.perturbation_scale,
                    steering_config.perturbation_strategy
                )
                
                # Evaluate
                evaluation = self.evaluator.evaluate_single_steering(
                    original_text=baseline_text,
                    perturbed_text=steered_text,
                    target_dimension=steering_config.target_dimension,
                    steering_magnitude=steering_config.perturbation_scale,
                    layer=steering_config.layer,
                    pc_axis=steering_config.pc_axis
                )
                
                results.append(evaluation)
                
                # Print brief result
                print(f"    Dimension shift: {evaluation.dimension_shift:.3f}, "
                      f"Semantic sim: {evaluation.semantic_similarity:.3f}, "
                      f"Fluency: {evaluation.fluency_score:.3f}")
                
            except Exception as e:
                print(f"    Error processing prompt: {e}")
                continue
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def run_comprehensive_experiments(
        self,
        manifold_results: Dict[str, Dict[int, ManifoldAnalysis]],
        evaluation_prompts: Dict[str, List[str]]
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Run comprehensive steering experiments across all configurations.
        
        Args:
            manifold_results: Results from manifold analysis
            evaluation_prompts: Test prompts for each concept
            
        Returns:
            Dictionary mapping experiment IDs to evaluation results
        """
        all_results = {}
        experiment_id = 0
        
        # Iterate through all combinations
        for concept, layer_results in manifold_results.items():
            if concept not in evaluation_prompts:
                print(f"Warning: No evaluation prompts for concept {concept}")
                continue
            
            for layer, manifold_analysis in layer_results.items():
                
                # Only test effective PCs
                for pc_axis in manifold_analysis.effective_pcs[:3]:  # Test top 3 PCs
                    
                    # Test each semantic dimension as target
                    for target_dimension in ['emotional_valence', 'formality', 'specificity']:
                        
                        # Test each perturbation strategy
                        for strategy in self.config.perturbation_strategies:
                            
                            # Test multiple scales
                            for scale in self.config.perturbation_scales:
                                if scale == 0.0:
                                    continue  # Skip zero perturbation
                                
                                steering_config = SteeringConfig(
                                    concept=concept,
                                    layer=layer,
                                    pc_axis=pc_axis,
                                    target_dimension=target_dimension,
                                    perturbation_strategy=strategy,
                                    perturbation_scale=scale,
                                    evaluation_prompts=evaluation_prompts[concept]
                                )
                                
                                experiment_key = (
                                    f"{concept}_L{layer}_PC{pc_axis}_{target_dimension}_"
                                    f"{strategy}_scale{scale}"
                                )
                                
                                results = self.run_single_experiment(
                                    steering_config, manifold_analysis
                                )
                                
                                if results:
                                    all_results[experiment_key] = results
                                
                                experiment_id += 1
                                
                                # Break early for testing (remove for full experiments)
                                if experiment_id > 20:  # Limit for initial testing
                                    print("Stopping early for testing purposes...")
                                    return all_results
        
        return all_results
    
    def analyze_experiment_results(
        self,
        all_results: Dict[str, List[EvaluationResult]]
    ) -> Dict[str, Dict]:
        """
        Analyze the results of all experiments to identify successful steering.
        
        Args:
            all_results: Results from all experiments
            
        Returns:
            Comprehensive analysis including best performing experiments
        """
        analysis = {
            'experiment_summaries': {},
            'best_experiments': {},
            'overall_statistics': {}
        }
        
        # Analyze each experiment
        for experiment_key, results in all_results.items():
            if not results:
                continue
            
            # Compute statistics for this experiment
            dimension_shifts = [r.dimension_shift for r in results]
            semantic_similarities = [r.semantic_similarity for r in results]
            fluency_scores = [r.fluency_score for r in results]
            
            experiment_summary = {
                'num_results': len(results),
                'mean_dimension_shift': np.mean(dimension_shifts),
                'std_dimension_shift': np.std(dimension_shifts),
                'mean_semantic_similarity': np.mean(semantic_similarities),
                'mean_fluency': np.mean(fluency_scores),
                'success_rate': sum(
                    self.evaluator.check_success_criteria(r).get('target_dimension_shift', False)
                    for r in results
                ) / len(results)
            }
            
            analysis['experiment_summaries'][experiment_key] = experiment_summary
        
        # Find best experiments by different criteria
        if analysis['experiment_summaries']:
            # Best by dimension shift
            best_by_shift = max(
                analysis['experiment_summaries'].items(),
                key=lambda x: x[1]['mean_dimension_shift']
            )
            analysis['best_experiments']['highest_dimension_shift'] = best_by_shift
            
            # Best by success rate
            best_by_success = max(
                analysis['experiment_summaries'].items(),
                key=lambda x: x[1]['success_rate']
            )
            analysis['best_experiments']['highest_success_rate'] = best_by_success
            
            # Best balanced (dimension shift * semantic similarity)
            best_balanced = max(
                analysis['experiment_summaries'].items(),
                key=lambda x: x[1]['mean_dimension_shift'] * x[1]['mean_semantic_similarity']
            )
            analysis['best_experiments']['best_balanced'] = best_balanced
        
        # Overall statistics
        all_dimension_shifts = []
        all_semantic_similarities = []
        all_fluency_scores = []
        
        for results in all_results.values():
            all_dimension_shifts.extend([r.dimension_shift for r in results])
            all_semantic_similarities.extend([r.semantic_similarity for r in results])
            all_fluency_scores.extend([r.fluency_score for r in results])
        
        if all_dimension_shifts:
            analysis['overall_statistics'] = {
                'total_experiments': len(all_results),
                'total_evaluations': len(all_dimension_shifts),
                'overall_mean_dimension_shift': np.mean(all_dimension_shifts),
                'overall_mean_semantic_similarity': np.mean(all_semantic_similarities),
                'overall_mean_fluency': np.mean(all_fluency_scores),
                'successful_experiments': sum(
                    1 for summary in analysis['experiment_summaries'].values()
                    if summary['success_rate'] > 0.5
                )
            }
        
        return analysis
    
    def save_experiment_results(
        self,
        all_results: Dict[str, List[EvaluationResult]],
        analysis: Dict[str, Dict],
        filepath: str
    ):
        """Save comprehensive experiment results to JSON file"""
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
        
        # Convert results to serializable format
        serializable_results = {}
        
        for experiment_key, results in all_results.items():
            serializable_results[experiment_key] = []
            for result in results:
                result_dict = {
                    'original_text': result.original_text,
                    'perturbed_text': result.perturbed_text,
                    'semantic_similarity': convert_to_serializable(result.semantic_similarity),
                    'fluency_score': convert_to_serializable(result.fluency_score),
                    'dimension_scores': convert_to_serializable(result.dimension_scores),
                    'dimension_shift': convert_to_serializable(result.dimension_shift),
                    'target_dimension': result.target_dimension,
                    'steering_magnitude': convert_to_serializable(result.steering_magnitude),
                    'layer': convert_to_serializable(result.layer),
                    'pc_axis': convert_to_serializable(result.pc_axis)
                }
                serializable_results[experiment_key].append(result_dict)
        
        # Combine everything
        complete_results = {
            'experiment_results': serializable_results,
            'analysis': convert_to_serializable(analysis),
            'config': convert_to_serializable({
                'perturbation_scales': self.config.perturbation_scales,
                'perturbation_strategies': self.config.perturbation_strategies,
                'target_layers': self.config.target_layers,
                'target_concepts': self.config.target_concepts
            })
        }
        
        with open(filepath, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"Complete experiment results saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    from data_preparation import ConceptDatasetGenerator, create_evaluation_prompts
    from manifold_analysis import ConceptManifoldAnalyzer
    
    config = ExperimentConfig()
    
    # Generate test dataset and evaluation prompts
    generator = ConceptDatasetGenerator()
    dataset = generator.create_balanced_dataset(['dog'])
    eval_prompts = create_evaluation_prompts()
    
    # Analyze manifolds (would normally load from saved results)
    analyzer = ConceptManifoldAnalyzer(config)
    manifold_results = analyzer.analyze_all_concepts(dataset, target_layers=[16])
    
    # Run steering experiments
    experiment_runner = SteeringExperimentRunner(config)
    all_results = experiment_runner.run_comprehensive_experiments(
        manifold_results, eval_prompts
    )
    
    # Analyze results
    analysis = experiment_runner.analyze_experiment_results(all_results)
    
    # Save results
    experiment_runner.save_experiment_results(
        all_results, analysis, "steering_experiment_results.json"
    )
    
    print(f"\nExperiment complete! Analyzed {len(all_results)} experiment configurations.")
    if analysis['best_experiments']:
        best = analysis['best_experiments']['highest_dimension_shift']
        print(f"Best experiment by dimension shift: {best[0]} (shift: {best[1]['mean_dimension_shift']:.3f})") 