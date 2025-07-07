"""
Comprehensive evaluation metrics for concept manifold steering experiments.
Measures semantic preservation, fluency, target dimension shifts, and statistical significance.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import re
from scipy import stats
from dataclasses import dataclass
import json

from config import SEMANTIC_DIMENSIONS, SUCCESS_CRITERIA, DEVICE

@dataclass
class EvaluationResult:
    """Results of steering evaluation for a single test case"""
    original_text: str
    perturbed_text: str
    semantic_similarity: float
    fluency_score: float
    dimension_scores: Dict[str, float]
    dimension_shift: float
    target_dimension: str
    steering_magnitude: float
    layer: int
    pc_axis: int

class SemanticEvaluator:
    """Evaluates semantic properties of generated text"""
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension_keywords = SEMANTIC_DIMENSIONS
        
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using sentence embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = torch.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        ).item()
        return max(0.0, similarity)  # Ensure non-negative
    
    def compute_dimension_score(self, text: str, dimension: str) -> float:
        """
        Compute a score for a specific semantic dimension based on keyword presence.
        
        Args:
            text: Text to analyze
            dimension: Semantic dimension to score
            
        Returns:
            Score between -1 and 1 representing dimension alignment
        """
        if dimension not in self.dimension_keywords:
            return 0.0
        
        text_lower = text.lower()
        dimension_config = self.dimension_keywords[dimension]
        
        if dimension == 'emotional_valence':
            positive_count = sum(1 for kw in dimension_config['positive_keywords'] if kw in text_lower)
            negative_count = sum(1 for kw in dimension_config['negative_keywords'] if kw in text_lower)
            neutral_count = sum(1 for kw in dimension_config['neutral_keywords'] if kw in text_lower)
            
            total_count = positive_count + negative_count + neutral_count
            if total_count == 0:
                return 0.0
            
            # Score from -1 (negative) to +1 (positive), 0 for neutral
            score = (positive_count - negative_count) / total_count
            return score
            
        elif dimension == 'formality':
            formal_count = sum(1 for kw in dimension_config['formal_keywords'] if kw in text_lower)
            casual_count = sum(1 for kw in dimension_config['casual_keywords'] if kw in text_lower)
            
            total_count = formal_count + casual_count
            if total_count == 0:
                return 0.0
            
            # Score from -1 (casual) to +1 (formal)
            score = (formal_count - casual_count) / total_count
            return score
            
        elif dimension == 'specificity':
            specific_count = sum(1 for kw in dimension_config['specific_keywords'] if kw in text_lower)
            general_count = sum(1 for kw in dimension_config['general_keywords'] if kw in text_lower)
            
            total_count = specific_count + general_count
            if total_count == 0:
                return 0.0
            
            # Score from -1 (general) to +1 (specific)
            score = (specific_count - general_count) / total_count
            return score
            
        elif dimension == 'perspective':
            scientific_count = sum(1 for kw in dimension_config['scientific_keywords'] if kw in text_lower)
            personal_count = sum(1 for kw in dimension_config['personal_keywords'] if kw in text_lower)
            
            total_count = scientific_count + personal_count
            if total_count == 0:
                return 0.0
            
            # Score from -1 (personal) to +1 (scientific)
            score = (scientific_count - personal_count) / total_count
            return score
        
        return 0.0
    
    def compute_all_dimension_scores(self, text: str) -> Dict[str, float]:
        """Compute scores for all semantic dimensions"""
        return {
            dimension: self.compute_dimension_score(text, dimension)
            for dimension in self.dimension_keywords.keys()
        }

class FluencyEvaluator:
    """Evaluates fluency and naturalness of generated text"""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the language model for perplexity computation"""
        if self.model is None:
            print("Loading fluency evaluation model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map=DEVICE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text using the language model.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Perplexity score (lower is better)
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def compute_fluency_score(self, text: str) -> float:
        """
        Compute a normalized fluency score based on multiple factors.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Fluency score between 0 and 1 (higher is better)
        """
        # Basic length check
        if len(text.strip()) < 5:
            return 0.0
        
        # Perplexity component (normalized)
        perplexity = self.compute_perplexity(text)
        perplexity_score = max(0.0, 1.0 - (perplexity - 1.0) / 100.0)  # Normalize around reasonable range
        
        # Syntactic correctness (basic heuristics)
        syntax_score = self._compute_syntax_score(text)
        
        # Combine scores
        fluency_score = 0.6 * perplexity_score + 0.4 * syntax_score
        return max(0.0, min(1.0, fluency_score))
    
    def _compute_syntax_score(self, text: str) -> float:
        """Compute basic syntactic correctness score"""
        score = 1.0
        
        # Check for basic punctuation
        if not re.search(r'[.!?]$', text.strip()):
            score -= 0.2
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        if avg_sentence_length < 3:  # Very short sentences
            score -= 0.3
        elif avg_sentence_length > 50:  # Very long sentences
            score -= 0.2
        
        # Check for repeated words (indication of generation issues)
        words = text.lower().split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                score -= 0.4
        
        return max(0.0, score)

class SteeringEffectivenessEvaluator:
    """Evaluates the effectiveness of steering interventions"""
    
    def __init__(self):
        self.semantic_evaluator = SemanticEvaluator()
        self.fluency_evaluator = FluencyEvaluator()
    
    def evaluate_single_steering(
        self,
        original_text: str,
        perturbed_text: str,
        target_dimension: str,
        steering_magnitude: float,
        layer: int,
        pc_axis: int
    ) -> EvaluationResult:
        """
        Evaluate a single steering intervention.
        
        Args:
            original_text: Original generated text
            perturbed_text: Text generated with steering
            target_dimension: The semantic dimension being targeted
            steering_magnitude: Magnitude of the steering intervention
            layer: Layer where intervention was applied
            pc_axis: Principal component axis used
            
        Returns:
            EvaluationResult object with all metrics
        """
        # Semantic similarity
        semantic_similarity = self.semantic_evaluator.compute_semantic_similarity(
            original_text, perturbed_text
        )
        
        # Fluency
        fluency_score = self.fluency_evaluator.compute_fluency_score(perturbed_text)
        
        # Dimension scores
        original_scores = self.semantic_evaluator.compute_all_dimension_scores(original_text)
        perturbed_scores = self.semantic_evaluator.compute_all_dimension_scores(perturbed_text)
        
        # Dimension shift in target dimension
        dimension_shift = abs(
            perturbed_scores.get(target_dimension, 0.0) - 
            original_scores.get(target_dimension, 0.0)
        )
        
        return EvaluationResult(
            original_text=original_text,
            perturbed_text=perturbed_text,
            semantic_similarity=semantic_similarity,
            fluency_score=fluency_score,
            dimension_scores=perturbed_scores,
            dimension_shift=dimension_shift,
            target_dimension=target_dimension,
            steering_magnitude=steering_magnitude,
            layer=layer,
            pc_axis=pc_axis
        )
    
    def evaluate_steering_consistency(
        self,
        results: List[EvaluationResult],
        target_dimension: str
    ) -> Dict[str, float]:
        """
        Evaluate consistency of steering effects across multiple tests.
        
        Args:
            results: List of evaluation results for the same steering setup
            target_dimension: The target semantic dimension
            
        Returns:
            Dictionary with consistency metrics
        """
        if not results:
            return {}
        
        # Extract dimension shifts
        dimension_shifts = [r.dimension_shift for r in results]
        semantic_similarities = [r.semantic_similarity for r in results]
        fluency_scores = [r.fluency_score for r in results]
        
        # Compute statistics
        consistency_metrics = {
            'mean_dimension_shift': np.mean(dimension_shifts),
            'std_dimension_shift': np.std(dimension_shifts),
            'mean_semantic_similarity': np.mean(semantic_similarities),
            'std_semantic_similarity': np.std(semantic_similarities),
            'mean_fluency': np.mean(fluency_scores),
            'std_fluency': np.std(fluency_scores),
            'consistency_score': 1.0 / (1.0 + np.std(dimension_shifts))  # Higher is more consistent
        }
        
        return consistency_metrics
    
    def check_success_criteria(self, result: EvaluationResult) -> Dict[str, bool]:
        """
        Check if a steering result meets the predefined success criteria.
        
        Args:
            result: EvaluationResult to check
            
        Returns:
            Dictionary mapping criteria names to pass/fail status
        """
        criteria_results = {}
        
        for criterion_name, criterion_config in SUCCESS_CRITERIA.items():
            if criterion_name == 'semantic_preservation':
                value = result.semantic_similarity
            elif criterion_name == 'target_dimension_shift':
                value = result.dimension_shift
            elif criterion_name == 'fluency_preservation':
                # For fluency, we want high scores, so we don't use ratio here
                value = result.fluency_score
                # Adjust threshold for direct fluency score
                threshold = 0.6  # Minimum acceptable fluency
                criteria_results[criterion_name] = value >= threshold
                continue
            else:
                continue  # Skip criteria that require multiple results
            
            threshold = criterion_config['threshold']
            direction = criterion_config['direction']
            
            if direction == 'greater_than':
                criteria_results[criterion_name] = value >= threshold
            elif direction == 'less_than':
                criteria_results[criterion_name] = value <= threshold
            else:
                criteria_results[criterion_name] = False
        
        return criteria_results

class StatisticalAnalyzer:
    """Performs statistical analysis of steering effects"""
    
    def __init__(self):
        pass
    
    def test_steering_significance(
        self,
        control_results: List[EvaluationResult],
        treatment_results: List[EvaluationResult],
        metric: str = 'dimension_shift'
    ) -> Dict[str, float]:
        """
        Test statistical significance of steering effects using t-test.
        
        Args:
            control_results: Results without steering (control condition)
            treatment_results: Results with steering (treatment condition)
            metric: Which metric to test ('dimension_shift', 'semantic_similarity', etc.)
            
        Returns:
            Dictionary with statistical test results
        """
        # Extract metric values
        if metric == 'dimension_shift':
            control_values = [r.dimension_shift for r in control_results]
            treatment_values = [r.dimension_shift for r in treatment_results]
        elif metric == 'semantic_similarity':
            control_values = [r.semantic_similarity for r in control_results]
            treatment_values = [r.semantic_similarity for r in treatment_results]
        elif metric == 'fluency_score':
            control_values = [r.fluency_score for r in control_results]
            treatment_values = [r.fluency_score for r in treatment_results]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_values) - 1) * np.var(control_values, ddof=1) +
             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) /
            (len(control_values) + len(treatment_values) - 2)
        )
        
        if pooled_std > 0:
            effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
        else:
            effect_size = 0.0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'significant': bool(p_value < SUCCESS_CRITERIA['statistical_significance']['threshold']),
            'control_mean': float(np.mean(control_values)),
            'treatment_mean': float(np.mean(treatment_values)),
            'control_std': float(np.std(control_values)),
            'treatment_std': float(np.std(treatment_values))
        }
    
    def analyze_dose_response(
        self,
        results_by_magnitude: Dict[float, List[EvaluationResult]],
        metric: str = 'dimension_shift'
    ) -> Dict[str, float]:
        """
        Analyze dose-response relationship between steering magnitude and effect.
        
        Args:
            results_by_magnitude: Dictionary mapping steering magnitudes to results
            metric: Which metric to analyze
            
        Returns:
            Dictionary with dose-response analysis results
        """
        magnitudes = []
        mean_effects = []
        
        for magnitude, results in results_by_magnitude.items():
            if not results:
                continue
                
            magnitudes.append(magnitude)
            
            if metric == 'dimension_shift':
                effects = [r.dimension_shift for r in results]
            elif metric == 'semantic_similarity':
                effects = [r.semantic_similarity for r in results]
            elif metric == 'fluency_score':
                effects = [r.fluency_score for r in results]
            else:
                effects = [0.0] * len(results)
            
            mean_effects.append(np.mean(effects))
        
        # Compute correlation
        if len(magnitudes) > 2:
            correlation, p_value = stats.pearsonr(magnitudes, mean_effects)
        else:
            correlation, p_value = 0.0, 1.0
        
        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'magnitudes': magnitudes,
            'mean_effects': mean_effects,
            'monotonic': bool(correlation > 0.3)  # Rough threshold for monotonic relationship
        }

def save_evaluation_results(
    results: List[EvaluationResult],
    consistency_metrics: Dict[str, Dict[str, float]],
    statistical_tests: Dict[str, Dict[str, float]],
    filepath: str
):
    """Save comprehensive evaluation results to JSON file"""
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
    
    # Convert EvaluationResult objects to dictionaries
    results_dict = []
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
        results_dict.append(result_dict)
    
    # Combine all results
    all_results = {
        'individual_results': results_dict,
        'consistency_metrics': convert_to_serializable(consistency_metrics),
        'statistical_tests': convert_to_serializable(statistical_tests),
        'summary': convert_to_serializable({
            'total_evaluations': len(results),
            'mean_semantic_similarity': np.mean([r.semantic_similarity for r in results]),
            'mean_fluency_score': np.mean([r.fluency_score for r in results]),
            'mean_dimension_shift': np.mean([r.dimension_shift for r in results])
        })
    }
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Evaluation results saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    evaluator = SteeringEffectivenessEvaluator()
    
    # Test evaluation on example texts
    original = "Dogs are animals that many people keep as pets."
    perturbed_positive = "Dogs are wonderful, amazing animals that bring incredible joy to families."
    perturbed_negative = "Dogs can be problematic, noisy animals that cause various difficulties."
    
    result_positive = evaluator.evaluate_single_steering(
        original, perturbed_positive, 'emotional_valence', 1.0, 16, 0
    )
    
    result_negative = evaluator.evaluate_single_steering(
        original, perturbed_negative, 'emotional_valence', -1.0, 16, 0
    )
    
    print(f"Positive steering result:")
    print(f"  Dimension shift: {result_positive.dimension_shift:.3f}")
    print(f"  Semantic similarity: {result_positive.semantic_similarity:.3f}")
    print(f"  Fluency score: {result_positive.fluency_score:.3f}")
    
    print(f"\nNegative steering result:")
    print(f"  Dimension shift: {result_negative.dimension_shift:.3f}")
    print(f"  Semantic similarity: {result_negative.semantic_similarity:.3f}")
    print(f"  Fluency score: {result_negative.fluency_score:.3f}") 