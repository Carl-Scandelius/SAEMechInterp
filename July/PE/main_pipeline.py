"""
Main pipeline for concept manifold steering experiments.
Orchestrates the complete research study from data preparation to final analysis.
"""

import os
import json
import argparse
from datetime import datetime
import torch

from config import ExperimentConfig
from data_preparation import ConceptDatasetGenerator, create_evaluation_prompts
from manifold_analysis import ConceptManifoldAnalyzer, validate_manifold_quality
from steering_experiments import SteeringExperimentRunner
from evaluation_metrics import save_evaluation_results, StatisticalAnalyzer

def create_experiment_directory(base_name: str = "steering_experiment") -> str:
    """Create a timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{base_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_config(config: ExperimentConfig, exp_dir: str):
    """Save experiment configuration"""
    config_dict = {
        'model_name': config.model_name,
        'device': config.device,
        'target_layers': config.target_layers,
        'target_concepts': config.target_concepts,
        'perturbation_scales': config.perturbation_scales,
        'perturbation_strategies': config.perturbation_strategies,
        'max_new_tokens': config.max_new_tokens,
        'temperature': config.temperature,
        'do_sample': config.do_sample,
        'top_p': config.top_p,
        'num_repetitions': config.num_repetitions,
        'significance_threshold': config.significance_threshold,
        'effect_size_threshold': config.effect_size_threshold
    }
    
    with open(os.path.join(exp_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

def run_complete_pipeline(
    config: ExperimentConfig,
    exp_dir: str,
    skip_data_generation: bool = False,
    skip_manifold_analysis: bool = False,
    skip_steering_experiments: bool = False
) -> dict:
    """
    Run the complete experimental pipeline.
    
    Args:
        config: Experiment configuration
        exp_dir: Experiment directory
        skip_data_generation: Skip data generation if files exist
        skip_manifold_analysis: Skip manifold analysis if files exist
        skip_steering_experiments: Skip steering experiments if files exist
        
    Returns:
        Dictionary with pipeline results and file paths
    """
    results = {
        'experiment_directory': exp_dir,
        'data_files': {},
        'manifold_files': {},
        'steering_files': {},
        'analysis_files': {}
    }
    
    print("="*80)
    print("CONCEPT MANIFOLD STEERING EXPERIMENT PIPELINE")
    print("="*80)
    
    # Step 1: Data Preparation
    print("\n1. DATA PREPARATION")
    print("-" * 40)
    
    dataset_file = os.path.join(exp_dir, 'semantic_concept_dataset.json')
    eval_prompts_file = os.path.join(exp_dir, 'evaluation_prompts.json')
    
    if not skip_data_generation or not os.path.exists(dataset_file):
        print("Generating semantic concept dataset...")
        generator = ConceptDatasetGenerator()
        dataset = generator.create_balanced_dataset(config.target_concepts)
        
        print("Validating dataset balance...")
        validation = generator.validate_dataset_balance(dataset)
        print(f"Dataset validation: {validation['passes_validation']}")
        print(f"Total samples: {validation['total_samples']}")
        print(f"Balance score: {validation['balance_score']:.3f}")
        
        generator.save_dataset(dataset, dataset_file)
        
        print("Creating evaluation prompts...")
        eval_prompts = create_evaluation_prompts()
        with open(eval_prompts_file, 'w') as f:
            json.dump(eval_prompts, f, indent=2)
        
        # Save validation results
        validation_file = os.path.join(exp_dir, 'dataset_validation.json')
        with open(validation_file, 'w') as f:
            json.dump(validation, f, indent=2)
        
        results['data_files']['dataset'] = dataset_file
        results['data_files']['evaluation_prompts'] = eval_prompts_file
        results['data_files']['validation'] = validation_file
    else:
        print("Loading existing dataset...")
        generator = ConceptDatasetGenerator()
        dataset = generator.load_dataset(dataset_file)
        with open(eval_prompts_file, 'r') as f:
            eval_prompts = json.load(f)
        results['data_files']['dataset'] = dataset_file
        results['data_files']['evaluation_prompts'] = eval_prompts_file
    
    # Step 2: Manifold Analysis
    print("\n2. MANIFOLD ANALYSIS")
    print("-" * 40)
    
    manifold_results_file = os.path.join(exp_dir, 'manifold_analysis_results.json')
    manifold_validation_file = os.path.join(exp_dir, 'manifold_validation.json')
    
    if not skip_manifold_analysis or not os.path.exists(manifold_results_file):
        print("Initializing manifold analyzer...")
        analyzer = ConceptManifoldAnalyzer(config)
        
        print("Analyzing concept manifolds...")
        manifold_results = analyzer.analyze_all_concepts(dataset, config.target_layers)
        
        print("Validating manifold quality...")
        manifold_validations = {}
        for concept, layer_results in manifold_results.items():
            manifold_validations[concept] = {}
            for layer, analysis in layer_results.items():
                validation = validate_manifold_quality(analysis)
                manifold_validations[concept][layer] = validation
                
                print(f"  {concept} layer {layer}: "
                      f"sufficient_pcs={validation['sufficient_pcs']}, "
                      f"meaningful_variance={validation['meaningful_variance']}, "
                      f"stable_manifold={validation['stable_manifold']}")
        
        # Save results
        analyzer.save_analysis_results(manifold_results, manifold_results_file)
        with open(manifold_validation_file, 'w') as f:
            json.dump(manifold_validations, f, indent=2)
        
        results['manifold_files']['analysis'] = manifold_results_file
        results['manifold_files']['validation'] = manifold_validation_file
        
        # Clear GPU memory
        del analyzer
        torch.cuda.empty_cache()
    else:
        print("Loading existing manifold analysis...")
        # Note: For full pipeline, we'd need to reload ManifoldAnalysis objects
        # For now, we'll assume steering experiments will reload as needed
        results['manifold_files']['analysis'] = manifold_results_file
        results['manifold_files']['validation'] = manifold_validation_file
    
    # Step 3: Steering Experiments
    print("\n3. STEERING EXPERIMENTS")
    print("-" * 40)
    
    steering_results_file = os.path.join(exp_dir, 'steering_experiment_results.json')
    
    if not skip_steering_experiments or not os.path.exists(steering_results_file):
        print("Initializing steering experiment runner...")
        experiment_runner = SteeringExperimentRunner(config)
        
        # For the pipeline, we need to reload manifold results as objects
        # This is a simplified version - in practice, you'd implement proper serialization
        print("Note: Reloading manifold analysis for steering experiments...")
        analyzer = ConceptManifoldAnalyzer(config)
        manifold_results = analyzer.analyze_all_concepts(dataset, config.target_layers)
        
        print("Running steering experiments...")
        all_steering_results = experiment_runner.run_comprehensive_experiments(
            manifold_results, eval_prompts
        )
        
        print("Analyzing experiment results...")
        steering_analysis = experiment_runner.analyze_experiment_results(all_steering_results)
        
        # Save results
        experiment_runner.save_experiment_results(
            all_steering_results, steering_analysis, steering_results_file
        )
        
        results['steering_files']['experiments'] = steering_results_file
        
        # Clear GPU memory
        del experiment_runner, analyzer
        torch.cuda.empty_cache()
    else:
        print("Loading existing steering experiment results...")
        results['steering_files']['experiments'] = steering_results_file
    
    # Step 4: Statistical Analysis
    print("\n4. STATISTICAL ANALYSIS")
    print("-" * 40)
    
    statistical_analysis_file = os.path.join(exp_dir, 'statistical_analysis.json')
    
    print("Performing statistical analysis...")
    # Load steering results for analysis
    with open(steering_results_file, 'r') as f:
        steering_data = json.load(f)
    
    # Extract key findings
    statistical_summary = {
        'total_experiments': len(steering_data['experiment_results']),
        'successful_experiments': steering_data['analysis']['overall_statistics'].get('successful_experiments', 0),
        'best_experiments': steering_data['analysis']['best_experiments'],
        'overall_statistics': steering_data['analysis']['overall_statistics']
    }
    
    # Add success rate analysis
    experiment_summaries = steering_data['analysis']['experiment_summaries']
    success_rates = [summary['success_rate'] for summary in experiment_summaries.values()]
    
    if success_rates:
        statistical_summary['success_rate_statistics'] = {
            'mean_success_rate': sum(success_rates) / len(success_rates),
            'max_success_rate': max(success_rates),
            'experiments_with_50plus_success': sum(1 for rate in success_rates if rate >= 0.5),
            'experiments_with_75plus_success': sum(1 for rate in success_rates if rate >= 0.75)
        }
    
    with open(statistical_analysis_file, 'w') as f:
        json.dump(statistical_summary, f, indent=2)
    
    results['analysis_files']['statistical'] = statistical_analysis_file
    
    # Step 5: Generate Summary Report
    print("\n5. GENERATING SUMMARY REPORT")
    print("-" * 40)
    
    summary_report = generate_summary_report(results, statistical_summary)
    summary_file = os.path.join(exp_dir, 'experiment_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    results['analysis_files']['summary'] = summary_file
    
    print(f"\nPipeline complete! Results saved in: {exp_dir}")
    print(f"Summary report: {summary_file}")
    
    return results

def generate_summary_report(results: dict, statistical_summary: dict) -> str:
    """Generate a human-readable summary report"""
    report = []
    report.append("CONCEPT MANIFOLD STEERING EXPERIMENT SUMMARY")
    report.append("=" * 60)
    report.append(f"Experiment directory: {results['experiment_directory']}")
    report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data preparation summary
    report.append("DATA PREPARATION:")
    report.append(f"  Dataset file: {os.path.basename(results['data_files']['dataset'])}")
    report.append(f"  Evaluation prompts: {os.path.basename(results['data_files']['evaluation_prompts'])}")
    report.append("")
    
    # Manifold analysis summary
    report.append("MANIFOLD ANALYSIS:")
    report.append(f"  Analysis results: {os.path.basename(results['manifold_files']['analysis'])}")
    report.append(f"  Validation results: {os.path.basename(results['manifold_files']['validation'])}")
    report.append("")
    
    # Steering experiments summary
    report.append("STEERING EXPERIMENTS:")
    report.append(f"  Total experiments: {statistical_summary['total_experiments']}")
    report.append(f"  Successful experiments: {statistical_summary['successful_experiments']}")
    
    if 'success_rate_statistics' in statistical_summary:
        stats = statistical_summary['success_rate_statistics']
        report.append(f"  Mean success rate: {stats['mean_success_rate']:.3f}")
        report.append(f"  Max success rate: {stats['max_success_rate']:.3f}")
        report.append(f"  Experiments with ≥50% success: {stats['experiments_with_50plus_success']}")
        report.append(f"  Experiments with ≥75% success: {stats['experiments_with_75plus_success']}")
    report.append("")
    
    # Best experiments
    if 'best_experiments' in statistical_summary:
        report.append("BEST PERFORMING EXPERIMENTS:")
        best = statistical_summary['best_experiments']
        
        if 'highest_dimension_shift' in best:
            exp_name, exp_data = best['highest_dimension_shift']
            report.append(f"  Highest dimension shift: {exp_name}")
            report.append(f"    Mean dimension shift: {exp_data['mean_dimension_shift']:.3f}")
            report.append(f"    Success rate: {exp_data['success_rate']:.3f}")
        
        if 'highest_success_rate' in best:
            exp_name, exp_data = best['highest_success_rate']
            report.append(f"  Highest success rate: {exp_name}")
            report.append(f"    Success rate: {exp_data['success_rate']:.3f}")
            report.append(f"    Mean dimension shift: {exp_data['mean_dimension_shift']:.3f}")
        
        if 'best_balanced' in best:
            exp_name, exp_data = best['best_balanced']
            report.append(f"  Best balanced: {exp_name}")
            report.append(f"    Mean dimension shift: {exp_data['mean_dimension_shift']:.3f}")
            report.append(f"    Mean semantic similarity: {exp_data['mean_semantic_similarity']:.3f}")
    
    report.append("")
    report.append("FILES GENERATED:")
    for category, files in results.items():
        if isinstance(files, dict):
            report.append(f"  {category.upper()}:")
            for file_type, filepath in files.items():
                report.append(f"    {file_type}: {os.path.basename(filepath)}")
    
    return "\n".join(report)

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Run concept manifold steering experiments')
    parser.add_argument('--concepts', nargs='+', default=['dog', 'cat', 'car', 'book'],
                       help='Concepts to analyze')
    parser.add_argument('--layers', nargs='+', type=int, default=[4, 8, 16, 24, 31],
                       help='Layers to analyze')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data generation if files exist')
    parser.add_argument('--skip-manifold', action='store_true',
                       help='Skip manifold analysis if files exist')
    parser.add_argument('--skip-steering', action='store_true',
                       help='Skip steering experiments if files exist')
    parser.add_argument('--exp-name', default='steering_experiment',
                       help='Base name for experiment directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig()
    config.target_concepts = args.concepts
    config.target_layers = args.layers
    
    # Create experiment directory
    exp_dir = create_experiment_directory(args.exp_name)
    
    # Save configuration
    save_config(config, exp_dir)
    
    print(f"Starting experiment with config:")
    print(f"  Concepts: {config.target_concepts}")
    print(f"  Layers: {config.target_layers}")
    print(f"  Device: {config.device}")
    print(f"  Experiment directory: {exp_dir}")
    
    # Run pipeline
    try:
        results = run_complete_pipeline(
            config=config,
            exp_dir=exp_dir,
            skip_data_generation=args.skip_data,
            skip_manifold_analysis=args.skip_manifold,
            skip_steering_experiments=args.skip_steering
        )
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results directory: {results['experiment_directory']}")
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 