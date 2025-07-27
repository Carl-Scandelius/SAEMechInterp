# Representation Lensing Experiment

## Overview
This experiment implements representation lensing on Llama 3.1-8B (non-instruct) to analyze how embeddings before target words affect next-token predictions.

## Target Words
- `animals`
- `furniture` 
- `food`

## Dataset
Loads sentences from `../tools/manifold_sentences_hard_exactword_1000.json`

## Experiment Steps

### 1. Embedding Extraction
For each target word and each layer in Llama 3.1-8B:
- Find the token immediately before the target word's first token
- Extract the activation embedding from that position
- Ensure we're not extracting from space tokens or special tokens

### 2. Baseline Next-Token Prediction
- Pass embeddings through the language modeling head
- Apply softmax to get probability distributions
- Store top 3 most probable next tokens and their probabilities
- Average probabilities across all sentences for each target word per layer

### 3. Local PCA Analysis
- For each target word and layer, perform PCA on embeddings
- Center embeddings by their mean (local PCA)
- Extract principal components and explained variance ratios

### 4. Perturbation Analysis
- Perturb embeddings along the top principal component
- Use perturbation factors: [0.5, 1.0, 2.0] × eigenvalue
- Repeat next-token prediction on perturbed embeddings

### 5. Results Generation
- Cosine similarity matrix between all principal components
- Top PC explained variance for each layer and target
- Average top 3 next-token predictions for baseline and perturbed conditions

## Usage

```bash
# Quick test
python core_experiment.py --num-sentences 10 --layers 0 15

# Full experiment
python core_experiment.py --layers 0 5 15 25 31

# All sentences (default)
python core_experiment.py
```

## Output
- `../prediction_steering_results/experiment_results.json`: Complete results
- `../prediction_steering_results/pc_similarity_matrix.png`: Cosine similarity heatmap

## Requirements
- PyTorch
- Transformers 
- scikit-learn
- matplotlib
- seaborn
- numpy 