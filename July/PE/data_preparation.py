"""
Data preparation module for concept manifold steering experiments.
Generates semantically-structured datasets with balanced representation across dimensions.
"""

import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from config import SEMANTIC_DIMENSIONS, PROMPT_TEMPLATES, MANIFOLD_VALIDATION_CRITERIA

@dataclass 
class SemanticDataPoint:
    """Represents a single data point with semantic annotations"""
    prompt: str
    concept: str
    semantic_dimension: str
    semantic_value: str  # e.g., 'positive', 'negative', 'formal', 'casual'
    keywords_present: List[str]

class ConceptDatasetGenerator:
    """Generates balanced concept datasets for manifold analysis"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def generate_semantic_concept_prompts(self, concept: str, num_samples_per_dimension: int = 25) -> Dict[str, List[str]]:
        """
        Generate semantically diverse prompts for a given concept.
        
        Args:
            concept: The target concept (e.g., 'dog', 'cat')
            num_samples_per_dimension: Number of samples to generate per semantic dimension
            
        Returns:
            Dictionary mapping semantic categories to lists of prompts
        """
        prompts_by_category = {}
        
        # Emotional valence prompts
        prompts_by_category['emotional_positive'] = self._generate_emotional_prompts(
            concept, 'positive', num_samples_per_dimension
        )
        prompts_by_category['emotional_negative'] = self._generate_emotional_prompts(
            concept, 'negative', num_samples_per_dimension
        )
        prompts_by_category['emotional_neutral'] = self._generate_emotional_prompts(
            concept, 'neutral', num_samples_per_dimension
        )
        
        # Formality prompts
        prompts_by_category['formal'] = self._generate_formality_prompts(
            concept, 'formal', num_samples_per_dimension
        )
        prompts_by_category['casual'] = self._generate_formality_prompts(
            concept, 'casual', num_samples_per_dimension
        )
        
        # Specificity prompts
        prompts_by_category['specific'] = self._generate_specificity_prompts(
            concept, 'specific', num_samples_per_dimension
        )
        prompts_by_category['general'] = self._generate_specificity_prompts(
            concept, 'general', num_samples_per_dimension
        )
        
        # Perspective prompts
        prompts_by_category['scientific'] = self._generate_perspective_prompts(
            concept, 'scientific', num_samples_per_dimension
        )
        prompts_by_category['personal'] = self._generate_perspective_prompts(
            concept, 'personal', num_samples_per_dimension
        )
        
        return prompts_by_category
    
    def _generate_emotional_prompts(self, concept: str, valence: str, num_samples: int) -> List[str]:
        """Generate prompts with specific emotional valence"""
        base_templates = [
            f"Describe {concept}s in a way that emphasizes their {{emotion}}.",
            f"Tell me about the {{emotion}} aspects of {concept}s.",
            f"What makes {concept}s {{emotion}}?",
            f"Explain why {concept}s can be {{emotion}}.",
            f"Discuss the {{emotion}} qualities of {concept}s."
        ]
        
        if valence == 'positive':
            emotions = ['wonderful', 'amazing', 'delightful', 'fantastic', 'excellent', 'beautiful', 'joyful']
        elif valence == 'negative':
            emotions = ['problematic', 'concerning', 'difficult', 'challenging', 'troublesome', 'annoying']
        else:  # neutral
            emotions = ['typical', 'normal', 'standard', 'ordinary', 'regular', 'average', 'common']
        
        prompts = []
        for _ in range(num_samples):
            template = random.choice(base_templates)
            emotion = random.choice(emotions)
            prompts.append(template.format(emotion=emotion))
        
        return prompts
    
    def _generate_formality_prompts(self, concept: str, style: str, num_samples: int) -> List[str]:
        """Generate prompts with specific formality level"""
        if style == 'formal':
            templates = [
                f"Provide a comprehensive academic analysis of {concept}s.",
                f"Examine the characteristics of {concept}s from a scholarly perspective.",
                f"Discuss the fundamental properties of {concept}s in formal terms.",
                f"Present a systematic overview of {concept}s and their attributes.",
                f"Analyze the distinctive features of {concept}s in an academic context."
            ]
        else:  # casual
            templates = [
                f"So, what's the deal with {concept}s?",
                f"Tell me about {concept}s in a chill way.",
                f"What's cool about {concept}s?",
                f"Give me the lowdown on {concept}s.",
                f"What makes {concept}s awesome?"
            ]
        
        prompts = []
        for _ in range(num_samples):
            prompts.append(random.choice(templates))
        
        return prompts
    
    def _generate_specificity_prompts(self, concept: str, level: str, num_samples: int) -> List[str]:
        """Generate prompts with specific detail level"""
        if level == 'specific':
            templates = [
                f"Provide detailed, specific information about {concept}s.",
                f"Explain precisely what characterizes {concept}s.",
                f"Give me thorough, specific details about {concept}s.",
                f"Describe exactly what makes {concept}s unique.",
                f"Provide comprehensive, detailed information about {concept}s."
            ]
        else:  # general
            templates = [
                f"Give me a general overview of {concept}s.",
                f"Broadly speaking, what are {concept}s like?",
                f"In general terms, describe {concept}s.",
                f"Provide a broad description of {concept}s.",
                f"Give me a general sense of what {concept}s are about."
            ]
        
        prompts = []
        for _ in range(num_samples):
            prompts.append(random.choice(templates))
        
        return prompts
    
    def _generate_perspective_prompts(self, concept: str, perspective: str, num_samples: int) -> List[str]:
        """Generate prompts with specific perspective"""
        if perspective == 'scientific':
            templates = [
                f"From a scientific research perspective, analyze {concept}s.",
                f"What does scientific evidence tell us about {concept}s?",
                f"Based on research and studies, describe {concept}s.",
                f"From a biological/scientific viewpoint, explain {concept}s.",
                f"What do scientific investigations reveal about {concept}s?"
            ]
        else:  # personal
            templates = [
                f"In my personal experience with {concept}s, I would say...",
                f"From my personal perspective, {concept}s are...",
                f"I personally think {concept}s are...",
                f"Based on my own experience, {concept}s...",
                f"In my opinion, what makes {concept}s special is..."
            ]
        
        prompts = []
        for _ in range(num_samples):
            prompts.append(random.choice(templates))
        
        return prompts
    
    def create_balanced_dataset(self, concepts: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Create a balanced dataset across all concepts and semantic dimensions.
        
        Args:
            concepts: List of concepts to generate data for
            
        Returns:
            Nested dictionary: {concept: {semantic_category: [prompts]}}
        """
        dataset = {}
        
        for concept in concepts:
            print(f"Generating prompts for concept: {concept}")
            dataset[concept] = self.generate_semantic_concept_prompts(concept)
            
        return dataset
    
    def validate_dataset_balance(self, dataset: Dict[str, Dict[str, List[str]]]) -> Dict[str, any]:
        """
        Validate that the dataset is properly balanced across semantic dimensions.
        
        Args:
            dataset: The generated dataset
            
        Returns:
            Dictionary with validation metrics
        """
        validation_results = {
            'concepts': list(dataset.keys()),
            'total_samples': 0,
            'samples_per_concept': {},
            'samples_per_dimension': {},
            'balance_score': 0.0,
            'passes_validation': True,
            'issues': []
        }
        
        # Count samples
        total_samples = 0
        for concept, categories in dataset.items():
            concept_samples = sum(len(prompts) for prompts in categories.values())
            validation_results['samples_per_concept'][concept] = concept_samples
            total_samples += concept_samples
            
            # Check minimum samples per concept
            if concept_samples < MANIFOLD_VALIDATION_CRITERIA['min_samples_per_concept']:
                validation_results['issues'].append(
                    f"Concept '{concept}' has only {concept_samples} samples, "
                    f"minimum required: {MANIFOLD_VALIDATION_CRITERIA['min_samples_per_concept']}"
                )
                validation_results['passes_validation'] = False
        
        validation_results['total_samples'] = total_samples
        
        # Check dimension balance
        all_dimensions = set()
        for concept, categories in dataset.items():
            all_dimensions.update(categories.keys())
        
        for dimension in all_dimensions:
            dimension_counts = []
            for concept in dataset:
                if dimension in dataset[concept]:
                    dimension_counts.append(len(dataset[concept][dimension]))
                else:
                    dimension_counts.append(0)
            
            validation_results['samples_per_dimension'][dimension] = {
                'total': sum(dimension_counts),
                'per_concept': dict(zip(dataset.keys(), dimension_counts)),
                'std_dev': np.std(dimension_counts)
            }
        
        # Calculate balance score (lower standard deviation = better balance)
        concept_counts = list(validation_results['samples_per_concept'].values())
        validation_results['balance_score'] = 1.0 / (1.0 + np.std(concept_counts))
        
        return validation_results
    
    def save_dataset(self, dataset: Dict[str, Dict[str, List[str]]], filepath: str):
        """Save the generated dataset to a JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, Dict[str, List[str]]]:
        """Load a dataset from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

def create_evaluation_prompts() -> Dict[str, List[str]]:
    """
    Create evaluation prompts separate from training data for unbiased testing.
    
    Returns:
        Dictionary mapping concepts to lists of evaluation prompts
    """
    evaluation_prompts = {}
    
    for concept in ['dog', 'cat', 'car', 'book']:
        evaluation_prompts[concept] = [
            f"Tell me about {concept}s.",
            f"Describe {concept}s.",
            f"What are {concept}s like?",
            f"Explain {concept}s to me.",
            f"Give me information about {concept}s."
        ]
    
    return evaluation_prompts

if __name__ == "__main__":
    # Generate and validate dataset
    generator = ConceptDatasetGenerator()
    concepts = ['dog', 'cat', 'car', 'book']
    
    print("Generating balanced semantic dataset...")
    dataset = generator.create_balanced_dataset(concepts)
    
    print("\nValidating dataset balance...")
    validation = generator.validate_dataset_balance(dataset)
    
    print(f"Dataset validation results:")
    print(f"- Total samples: {validation['total_samples']}")
    print(f"- Balance score: {validation['balance_score']:.3f}")
    print(f"- Passes validation: {validation['passes_validation']}")
    
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    # Save dataset
    generator.save_dataset(dataset, "semantic_concept_dataset.json")
    
    # Create evaluation prompts
    eval_prompts = create_evaluation_prompts()
    with open("evaluation_prompts.json", 'w') as f:
        json.dump(eval_prompts, f, indent=2)
    
    print("\nDataset generation complete!") 