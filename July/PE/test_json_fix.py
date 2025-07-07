#!/usr/bin/env python3
"""
Test script to verify JSON serialization fixes work correctly.
This tests the key serialization functions without running full experiments.
"""

import numpy as np
import torch
import json
import tempfile
import os

def test_manifold_analysis_serialization():
    """Test manifold analysis JSON serialization"""
    from manifold_analysis import ConceptManifoldAnalyzer
    from config import ExperimentConfig
    
    print("Testing manifold analysis serialization...")
    
    # Create mock data with numpy/torch types
    mock_results = {
        'dog': {
            16: type('MockAnalysis', (), {
                'concept': 'dog',
                'layer': 16,
                'eigenvalues': torch.tensor([1.5, 1.2, 0.8], dtype=torch.float32),
                'explained_variance_ratio': torch.tensor([0.4, 0.3, 0.2], dtype=torch.float32),
                'effective_pcs': [0, 1, 2],
                'semantic_correlations': {
                    'PC0_semantic_separation': np.float32(0.75),
                    'PC0_category_means': {
                        'positive': np.float64(1.2),
                        'negative': np.float32(-0.8)
                    }
                },
                'cross_validation_stability': np.float64(0.85),
                'concept_centroid': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
            })()
        }
    }
    
    config = ExperimentConfig()
    analyzer = ConceptManifoldAnalyzer(config)
    
    # Test serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        try:
            analyzer.save_analysis_results(mock_results, f.name)
            
            # Verify we can read it back
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
                print(f"✓ Manifold analysis serialization successful")
                print(f"  Sample data: {list(data.keys())}")
                
        finally:
            os.unlink(f.name)

def test_evaluation_metrics_serialization():
    """Test evaluation metrics JSON serialization"""
    from evaluation_metrics import EvaluationResult, save_evaluation_results
    
    print("Testing evaluation metrics serialization...")
    
    # Create mock results with numpy types
    mock_results = [
        EvaluationResult(
            original_text="Original text",
            perturbed_text="Perturbed text",
            semantic_similarity=np.float32(0.85),
            fluency_score=np.float64(0.92),
            dimension_scores={'emotional_valence': np.float32(0.3)},
            dimension_shift=np.float32(0.4),
            target_dimension='emotional_valence',
            steering_magnitude=np.float32(1.0),
            layer=np.int32(16),
            pc_axis=np.int32(0)
        )
    ]
    
    mock_consistency = {
        'test_experiment': {
            'mean_shift': np.float64(0.4),
            'std_shift': np.float32(0.1)
        }
    }
    
    mock_stats = {
        'significance_test': {
            't_statistic': np.float64(2.5),
            'p_value': np.float32(0.02),
            'correlation': np.float64(0.75)
        }
    }
    
    # Test serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        try:
            save_evaluation_results(mock_results, mock_consistency, mock_stats, f.name)
            
            # Verify we can read it back
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
                print(f"✓ Evaluation metrics serialization successful")
                print(f"  Result count: {len(data['individual_results'])}")
                
        finally:
            os.unlink(f.name)

def test_steering_experiments_serialization():
    """Test steering experiments JSON serialization"""
    from steering_experiments import SteeringExperimentRunner
    from evaluation_metrics import EvaluationResult
    from config import ExperimentConfig
    
    print("Testing steering experiments serialization...")
    
    # Create mock data
    mock_results = {
        'test_experiment': [
            EvaluationResult(
                original_text="Original",
                perturbed_text="Perturbed", 
                semantic_similarity=np.float32(0.8),
                fluency_score=np.float64(0.9),
                dimension_scores={'test': np.float32(0.5)},
                dimension_shift=np.float32(0.3),
                target_dimension='test',
                steering_magnitude=np.float32(1.0),
                layer=np.int32(16),
                pc_axis=np.int32(0)
            )
        ]
    }
    
    mock_analysis = {
        'experiment_summaries': {
            'test_exp': {
                'mean_dimension_shift': np.float64(0.3),
                'success_rate': np.float32(0.8)
            }
        },
        'overall_statistics': {
            'total_experiments': np.int32(1),
            'overall_mean_dimension_shift': np.float64(0.3)
        }
    }
    
    config = ExperimentConfig()
    runner = SteeringExperimentRunner(config)
    
    # Test serialization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        try:
            runner.save_experiment_results(mock_results, mock_analysis, f.name)
            
            # Verify we can read it back
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
                print(f"✓ Steering experiments serialization successful")
                print(f"  Experiment count: {len(data['experiment_results'])}")
                
        finally:
            os.unlink(f.name)

def main():
    """Run all serialization tests"""
    print("Running JSON serialization tests...\n")
    
    try:
        test_manifold_analysis_serialization()
        print()
        
        test_evaluation_metrics_serialization()
        print()
        
        test_steering_experiments_serialization()
        print()
        
        print("🎉 All JSON serialization tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 