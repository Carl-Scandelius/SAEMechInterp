"""
Configuration file for rigorous concept manifold steering experiments.
Defines all experimental parameters, semantic dimensions, and evaluation criteria.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Experimental design parameters
@dataclass
class ExperimentConfig:
    """Main experimental configuration"""
    # Model and data
    model_name: str = MODEL_NAME
    device: str = DEVICE
    torch_dtype = TORCH_DTYPE
    
    # Layer configuration - test across network depth
    target_layers: List[int] = None  # Will be set to [4, 8, 16, 24, 31] in __post_init__
    
    # Concept configuration
    target_concepts: List[str] = None  # Will be set in __post_init__
    
    # Generation parameters
    max_new_tokens: int = 50
    temperature: float = 0.0    # deterministic for sake of reproducibility
    do_sample: bool = False
    top_p: float = 0.9
    
    # Perturbation parameters
    perturbation_scales: List[float] = None  # Will be set in __post_init__
    perturbation_strategies: List[str] = None  # Will be set in __post_init__
    
    # Evaluation parameters
    num_test_prompts: int = 5
    num_repetitions: int = 3  # For statistical robustness
    
    # Statistical significance
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.3
    
    def __post_init__(self):
        if self.target_layers is None:
            # Strategic layer selection: early, middle, late
            self.target_layers = [4, 8, 16, 24, 31]
        
        if self.target_concepts is None:
            self.target_concepts = ['dog', 'cat', 'car', 'book']
        
        if self.perturbation_scales is None:
            # More focused range of perturbation scales
            self.perturbation_scales = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        
        if self.perturbation_strategies is None:
            self.perturbation_strategies = ['additive', 'projection_based', 'replacement']

# Semantic dimensions for evaluation
SEMANTIC_DIMENSIONS = {
    'emotional_valence': {
        'description': 'Positive vs negative emotional tone',
        'positive_keywords': ['wonderful', 'amazing', 'beautiful', 'joy', 'love', 'excellent', 'fantastic'],
        'negative_keywords': ['terrible', 'awful', 'horrible', 'hate', 'disgusting', 'awful', 'dreadful'],
        'neutral_keywords': ['normal', 'average', 'typical', 'standard', 'regular', 'ordinary']
    },
    'formality': {
        'description': 'Formal vs casual language style',
        'formal_keywords': ['furthermore', 'consequently', 'therefore', 'indeed', 'moreover', 'specifically'],
        'casual_keywords': ['like', 'you know', 'kinda', 'sorta', 'totally', 'awesome', 'cool']
    },
    'specificity': {
        'description': 'Specific detailed vs general abstract descriptions',
        'specific_keywords': ['exactly', 'precisely', 'specifically', 'particularly', 'detailed', 'thorough'],
        'general_keywords': ['generally', 'overall', 'broadly', 'typically', 'usually', 'often']
    },
    'perspective': {
        'description': 'Scientific vs personal perspective',
        'scientific_keywords': ['research', 'studies', 'evidence', 'analysis', 'investigation', 'scientific'],
        'personal_keywords': ['I think', 'in my opinion', 'personally', 'I believe', 'my experience', 'I feel']
    }
}

# Success criteria for steering effectiveness
SUCCESS_CRITERIA = {
    'semantic_preservation': {
        'metric': 'cosine_similarity',
        'threshold': 0.6,  # Maintain semantic relatedness
        'direction': 'greater_than'
    },
    'target_dimension_shift': {
        'metric': 'dimension_score_difference', 
        'threshold': 0.3,  # Meaningful change in target dimension
        'direction': 'greater_than'
    },
    'fluency_preservation': {
        'metric': 'perplexity_ratio',
        'threshold': 2.0,  # Perplexity shouldn't increase too much
        'direction': 'less_than'
    },
    'consistency_across_prompts': {
        'metric': 'steering_correlation',
        'threshold': 0.5,  # Consistent steering direction
        'direction': 'greater_than'
    },
    'statistical_significance': {
        'metric': 'p_value',
        'threshold': 0.05,
        'direction': 'less_than'
    }
}

# Test prompt templates for each concept
PROMPT_TEMPLATES = {
    'dog': {
        'base_prompts': [
            "Tell me about dogs.",
            "Describe dogs as animals.",
            "What are the characteristics of dogs?",
            "Explain what makes dogs unique.",
            "Discuss the nature of dogs."
        ],
        'semantic_targets': {
            'emotional_positive': "Tell me about dogs in a very positive way.",
            'emotional_negative': "Tell me about dogs focusing on potential problems.",
            'formal': "Provide a formal academic description of dogs.",
            'casual': "Tell me about dogs in a casual, friendly way.",
            'scientific': "Describe dogs from a scientific perspective.",
            'personal': "Tell me about dogs from your personal viewpoint."
        }
    },
    'cat': {
        'base_prompts': [
            "Tell me about cats.",
            "Describe cats as animals.",
            "What are the characteristics of cats?",
            "Explain what makes cats unique.",
            "Discuss the nature of cats."
        ],
        'semantic_targets': {
            'emotional_positive': "Tell me about cats in a very positive way.",
            'emotional_negative': "Tell me about cats focusing on potential problems.",
            'formal': "Provide a formal academic description of cats.",
            'casual': "Tell me about cats in a casual, friendly way.",
            'scientific': "Describe cats from a scientific perspective.",
            'personal': "Tell me about cats from your personal viewpoint."
        }
    },
    'car': {
        'base_prompts': [
            "Tell me about cars.",
            "Describe cars as vehicles.",
            "What are the characteristics of cars?",
            "Explain what makes cars useful.",
            "Discuss the role of cars."
        ],
        'semantic_targets': {
            'emotional_positive': "Tell me about cars in a very positive way.",
            'emotional_negative': "Tell me about cars focusing on potential problems.",
            'formal': "Provide a formal academic description of cars.",
            'casual': "Tell me about cars in a casual, friendly way.",
            'scientific': "Describe cars from a scientific perspective.",
            'personal': "Tell me about cars from your personal viewpoint."
        }
    },
    'book': {
        'base_prompts': [
            "Tell me about books.",
            "Describe books as objects.",
            "What are the characteristics of books?",
            "Explain what makes books valuable.",
            "Discuss the importance of books."
        ],
        'semantic_targets': {
            'emotional_positive': "Tell me about books in a very positive way.",
            'emotional_negative': "Tell me about books focusing on potential problems.",
            'formal': "Provide a formal academic description of books.",
            'casual': "Tell me about books in a casual, friendly way.",
            'scientific': "Describe books from a scientific perspective.",
            'personal': "Tell me about books from your personal viewpoint."
        }
    }
}

# System prompt - consistent across all experiments
SYSTEM_PROMPT = "You are a helpful assistant. Please respond naturally and helpfully."

# Validation criteria for concept manifolds
MANIFOLD_VALIDATION_CRITERIA = {
    'min_explained_variance': 0.05,  # Each PC should explain at least 5% of variance
    'max_pcs_to_analyze': 5,  # Focus on top 5 principal components
    'min_samples_per_concept': 20,  # Need sufficient data for robust PCA
    'cross_validation_folds': 3  # For validating manifold stability
} 