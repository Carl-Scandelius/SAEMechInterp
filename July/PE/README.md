# Concept Manifold Steering Experiments

A rigorous research framework for studying how perturbations to concept manifolds in language models can achieve controllable semantic steering of generated text.

## Overview

This research framework addresses the challenge of interpreting and controlling language model behavior through concept manifold analysis. Rather than relying on simple activation perturbations, we:

1. **Extract concept-specific manifolds** using semantically-structured datasets
2. **Validate manifold quality** through cross-validation and semantic correlation analysis  
3. **Test multiple steering strategies** with comprehensive evaluation metrics
4. **Measure effectiveness** through semantic preservation, fluency, and target dimension shifts

## Key Features

- **Semantic Dataset Generation**: Balanced prompts across emotional valence, formality, specificity, and perspective dimensions
- **Robust Manifold Analysis**: PCA with proper centering, stability validation, and semantic interpretation
- **Multiple Steering Strategies**: Additive, projection-based, and replacement perturbation methods
- **Comprehensive Evaluation**: Semantic similarity, fluency scores, dimension shift analysis, and statistical significance testing
- **Full Pipeline**: End-to-end automation from data preparation to final analysis

## Installation

```bash
# Clone or copy the PE directory
cd July/PE

# Install requirements
pip install -r requirements.txt

# Ensure you have access to the model (Llama 3.1 8B by default)
huggingface-cli login
```

## Quick Start

Run the complete experimental pipeline:

```bash
python main_pipeline.py
```

This will:
1. Generate semantically-structured datasets
2. Analyze concept manifolds across multiple layers
3. Run steering experiments with different strategies
4. Evaluate results and generate comprehensive reports

## Detailed Usage

### Custom Configuration

Modify `config.py` to adjust experimental parameters:

```python
# Target concepts to study
target_concepts = ['dog', 'cat', 'car', 'book']

# Model layers to analyze  
target_layers = [4, 8, 16, 24, 31]

# Perturbation scales to test
perturbation_scales = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

# Steering strategies
perturbation_strategies = ['additive', 'projection_based', 'replacement']
```

### Running Individual Components

#### 1. Data Preparation

```bash
python data_preparation.py
```

Generates:
- `semantic_concept_dataset.json`: Training prompts organized by semantic dimensions
- `evaluation_prompts.json`: Test prompts for unbiased evaluation

#### 2. Manifold Analysis

```bash
python manifold_analysis.py
```

Analyzes concept manifolds with:
- Proper centering (concept-specific, not global)
- Cross-validation stability testing
- Semantic correlation analysis
- Quality validation metrics

#### 3. Steering Experiments

```bash
python steering_experiments.py  
```

Tests multiple perturbation strategies:
- **Additive**: `activation += scale * pc_vector`
- **Projection-based**: `activation += scale * projection * normalized_pc`
- **Replacement**: Replace component in PC direction

#### 4. Evaluation

```bash
python evaluation_metrics.py
```

Computes comprehensive metrics:
- Semantic similarity (sentence embeddings)
- Fluency scores (perplexity + syntax)
- Dimension shifts (keyword-based scoring)
- Statistical significance testing

### Pipeline Options

```bash
# Skip existing steps
python main_pipeline.py --skip-data --skip-manifold

# Customize concepts and layers
python main_pipeline.py --concepts dog cat --layers 8 16 24

# Custom experiment name
python main_pipeline.py --exp-name emotional_steering_study
```

## Understanding Results

### Experiment Directory Structure

```
steering_experiment_YYYYMMDD_HHMMSS/
├── experiment_config.json              # Configuration used
├── semantic_concept_dataset.json       # Training data
├── evaluation_prompts.json            # Test prompts
├── dataset_validation.json            # Data quality metrics
├── manifold_analysis_results.json     # PCA results and manifold quality
├── manifold_validation.json           # Manifold quality assessments
├── steering_experiment_results.json   # Complete steering results
├── statistical_analysis.json          # Summary statistics
└── experiment_summary.txt             # Human-readable summary
```

### Key Metrics

#### Success Criteria
- **Semantic Preservation**: Cosine similarity ≥ 0.6
- **Target Dimension Shift**: Absolute change ≥ 0.3
- **Fluency Preservation**: Fluency score ≥ 0.6
- **Statistical Significance**: p-value < 0.05

#### Manifold Quality Indicators
- **Sufficient PCs**: ≥2 principal components explaining >5% variance each
- **Meaningful Variance**: Top PC explains >10% of variance
- **Stability**: Cross-validation similarity >0.3
- **Semantic Structure**: Clear separation between semantic categories

### Interpreting Results

1. **Best Experiments**: Look for high dimension shifts with preserved semantic similarity
2. **Consistency**: Low standard deviation across repetitions indicates reliable effects
3. **Layer Effects**: Compare effectiveness across different network depths
4. **Strategy Comparison**: Identify which perturbation method works best for each concept/dimension

## Research Questions Addressed

1. **Can concept manifolds enable controllable semantic steering?**
   - Measure: Success rate of experiments meeting all criteria

2. **Which perturbation strategies are most effective?** 
   - Compare: Additive vs projection-based vs replacement methods

3. **How does steering effectiveness vary across network layers?**
   - Analyze: Performance differences between early, middle, and late layers

4. **What semantic dimensions are most controllable?**
   - Evaluate: Emotional valence vs formality vs specificity vs perspective

5. **Do steering effects generalize across concepts?**
   - Test: Consistency of results across different base concepts

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch size in config
   - Use smaller model or fewer layers
   - Enable gradient checkpointing

2. **Poor Manifold Quality**
   - Increase number of training samples per dimension
   - Check dataset balance validation
   - Try different layer depths

3. **Weak Steering Effects**
   - Increase perturbation scales
   - Focus on layers with higher explained variance
   - Validate semantic dimension keywords

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate files:
- `dataset_validation.json`: Ensure balanced data
- `manifold_validation.json`: Verify manifold quality
- Individual experiment results in `steering_experiment_results.json`

## Extending the Framework

### Adding New Semantic Dimensions

1. Define keywords in `config.py`:
```python
SEMANTIC_DIMENSIONS['urgency'] = {
    'description': 'Urgent vs relaxed tone',
    'urgent_keywords': ['immediately', 'urgent', 'asap', 'critical'],
    'relaxed_keywords': ['eventually', 'whenever', 'leisurely', 'casual']
}
```

2. Update evaluation metrics in `evaluation_metrics.py`
3. Add prompt generation in `data_preparation.py`

### Testing New Models

1. Update `MODEL_NAME` in `config.py`
2. Adjust layer indices for different architectures
3. Validate activation extraction hooks work correctly

### Custom Perturbation Strategies

Add new strategies to `steering_experiments.py`:
```python
elif strategy == 'custom_strategy':
    # Implement your perturbation logic
    output[0][:, -1, :] = custom_perturbation(
        output[0][:, -1, :], perturbation_vector, perturbation_scale
    )
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{concept_manifold_steering,
  title={Concept Manifold Steering: Controllable Semantic Perturbations in Language Models},
  author={[Your Name]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 