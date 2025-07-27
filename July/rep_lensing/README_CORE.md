# Representation Lensing Experiment

## Overview

This experiment implements **representation lensing** to analyze how token embeddings just before target words can predict the target words through the language modeling head. We focus on three target words: **animals**, **furniture**, and **food**.

## Methodology

### Core Approach

1. **Target Word Selection**: We analyze three target concepts: `{animals, furniture, food}`
2. **Embedding Extraction**: For each sentence containing a target word, we extract the embedding of the token immediately **before** the target word's first token
3. **Prediction Analysis**: We pass these "pre-target" embeddings directly through Llama 3.1-8B's language modeling head to get next-token predictions
4. **Local PCA**: We perform Principal Component Analysis on embeddings for each target word, centering by the mean embedding for that target word at each layer
5. **Perturbation Experiments**: We perturb embeddings along the first Principal Component (PC1) with multipliers k ∈ {0.5, 1.0, 2.0} and analyze how predictions change

### Technical Details

- **Model**: Llama 3.1-8B (non-instruct version)
- **Layers Analyzed**: [0, 15, 31] (input, middle, output layers)  
- **Dataset Size**: All 1000 sentences per target word from manifold_sentences_hard_exactword_1000.json
- **Local PCA**: Centered on each target word's mean embedding per layer
- **Perturbation Formula**: `perturbed_embedding = centered_embedding + k * eigenvalue_1 * PC1`

## Key Research Questions

1. **Predictive Power**: How well can embeddings before target words predict the target words?
2. **Layer Differences**: How does predictive capability vary across transformer layers?
3. **Concept Separation**: How similar/different are the principal components across target concepts?
4. **Perturbation Effects**: How do controlled perturbations along PC1 affect next-token predictions?

## Expected Results

The experiment will provide:

1. **Average Top-3 Next-Token Predictions** for each target word and layer
2. **PC1 Explained Variance** for each target word and layer  
3. **Cosine Similarity Matrix** between all PC1s across target words and layers
4. **Perturbation Analysis** showing how predictions change when embeddings are modified along PC1

## File Structure

```
rep_lensing/
├── core_experiment.py          # Main experiment implementation
├── requirements_minimal.txt    # Python dependencies
├── README_CORE.md             # This file
└── rep_lensing_results/       # Generated results directory
    ├── experiment_results.json
    ├── pc1_cosine_similarity_matrix.png
    └── explained_variance_summary.png
```

## Usage

### Basic Usage

```bash
cd rep_lensing
python core_experiment.py
```

### Advanced Usage

```bash
# Analyze subset of sentences per target word
python core_experiment.py --num-sentences 500

# Analyze different layers
python core_experiment.py --layers 0 8 16 24 31

# Combine options
python core_experiment.py --num-sentences 200 --layers 0 15 31
```

## Implementation Classes

### `RepresentationLensingExtractor`
- Extracts embeddings from tokens immediately before target words
- Handles tokenization edge cases and target word detection
- Supports extraction from multiple transformer layers

### `LanguageModelingAnalyzer` 
- Passes embeddings through the language modeling head
- Computes softmax probabilities over vocabulary
- Returns top-k predictions averaged across all examples

### `LocalPCAAnalyzer`
- Performs PCA on embeddings centered by target word means
- Handles edge cases with insufficient data
- Returns principal components, eigenvalues, and explained variance

### `PerturbationAnalyzer`
- Perturbs embeddings along PC1 with specified multipliers
- Analyzes how perturbations affect next-token predictions
- Tests the hypothesis that PC1 captures target word direction

## Key Findings

The experiment will reveal:

1. **Layer-wise Prediction Accuracy**: Which layers best predict target words from preceding context
2. **Concept Clustering**: Whether different target concepts have similar or distinct principal components
3. **Perturbation Sensitivity**: How robust/sensitive predictions are to controlled modifications
4. **Embedding Geometry**: The geometric structure of the representation space around target concepts

## Future Extensions

- Scale to thousands of sentences per target word
- Add more diverse target concepts
- Analyze intermediate layers (every 4th layer)
- Test perturbations along multiple principal components
- Cross-linguistic analysis with multilingual models

## Dependencies

Install requirements:
```bash
pip install -r requirements_minimal.txt
```

**Note**: Requires access to Llama 3.1-8B model through HuggingFace Transformers. 