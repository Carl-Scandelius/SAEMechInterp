# Representation Lensing Experiment

## Overview
This experiment implements representation lensing on Llama 3.1-8B (non-instruct) to analyze how embeddings before target words affect next-token predictions.

## Target Words
- `animals`
- `furniture` 
- `food`

## Dataset
Loads sentences from `../tools/manifold_sentences_hard_exactword_1000.json`

**Sentence Filtering**: Only includes sentences where:
- The token before the target word predicts the target word as most likely next token
- **The prediction confidence is above 0.15** (15% probability threshold)

## Experiment Steps

### 1. Embedding Extraction
For each target word and each layer in Llama 3.1-8B:
- Find the token immediately before the target word's first token
- Extract the activation embedding from that position
- Ensure we're not extracting from space tokens or special tokens

### 2. Baseline Next-Token Prediction (Dual Projection Methods)
**Two projection methods are used to convert embeddings to token probabilities:**

**Method 1: Language Modeling Head (Standard)**
- Pass embeddings through the learned language modeling head: `lm_head(embeddings)`
- This is the standard approach used during model training

**Method 2: Embedding Transpose**
- Project using transpose of embedding matrix: `embeddings @ embed_weights.T`
- Uses the inverse transformation of the token embedding matrix
- Tests whether the embedding space is roughly "symmetric"

**For both methods:**
- Apply softmax to get probability distributions
- Store top 10 most probable next tokens and their probabilities
- Average probabilities across all sentences for each target word per layer

### 3. Local PCA Analysis
- For each target word and layer, perform PCA on embeddings
- Center embeddings by their mean (local PCA)
- Extract up to 100 principal components and explained variance ratios
- Compute eigenvalues for perturbation scaling

### 4. Multi-PC Perturbation Analysis (Both Projection Methods + Centroid)
- Perturb embeddings along multiple principal components: **1st, 2nd, 3rd, 4th, 5th, 10th, and 50th PC**
- **NEW: Centroid perturbations** using manifold centroid direction with PC1 eigenvalue magnitude
- Use perturbation factors: [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] × eigenvalue
- **Run perturbations with both LM head and embedding transpose projections**
- Compare how the same perturbations affect predictions differently depending on projection method
- Analyze how different principal components affect predictions differently

### 5. Layer 31 Dual Analysis (LM Head Only)
**Special analysis for the final transformer layer (31):**
- **Standard method**: Let embeddings pass through residual stream to LM head (normal forward pass)
- **Extracted method**: Extract layer 31 embeddings and pass directly to LM head (like other layers)
- Compare predictions between both methods to test consistency
- Compute cosine similarity between probability distributions
- Only performed when layer 31 is included and LM head projection is used

### 6. Additional Test Case Analysis (Both Projection Methods + Centroid)
Using the **same PCs computed from the original experiment**, test three additional input types:

**For each target word (animals, furniture, food):**
1. **Category word only**: `"animals"`, `"furniture"`, `"food"`
2. **Spelling format**: `"a-n-i-m-a-l-s spells "`, `"f-u-r-n-i-t-u-r-e spells "`, `"f-o-o-d spells "`
3. **Translation format**: `"Translated from German into English, 'Tiere' is "`, `"Translated from German into English, 'Möbel' is "`, `"Translated from German into English, 'Essen' is "`

**Analysis for each test case:**
- Extract embeddings from the **last token position** of each test case
- Apply the same PC-based perturbations from the original experiment
- **NEW: Apply centroid perturbations** using original experiment's manifold centroids
- **Test with both LM head and embedding transpose projections**
- Compare how projection method affects the same prompt types
- Analyze cross-prompt and cross-projection consistency

### 7. Results Generation
- **Cosine similarity matrices** between principal components (separate matrix for each PC tested, **2 decimal places**)
- PC explained variance for each layer and target
- **Dual projection results**: All analyses performed with both LM head and embedding transpose
- **Layer 31 dual analysis**: Standard vs extracted embedding comparison for final layer
- Top 10 next-token predictions for baseline and all perturbation conditions
- **Centroid perturbation results** alongside PC perturbations
- Results organized by projection method, PC index, target word, and layer
- **Cross-projection comparison**: Direct comparison of how projection method affects results
- **Additional test case results** organized by projection method, target word, layer, and case type

## Usage

```bash
# Full experiment with both projection methods (default)
python core_experiment.py

# Quick test with both projections
python core_experiment.py --num-sentences 10 --layers 0 15

# Test only LM head projection
python core_experiment.py --projection-methods lm_head

# Test only embedding transpose projection  
python core_experiment.py --projection-methods embedding_transpose

# Compare both methods explicitly
python core_experiment.py --projection-methods lm_head embedding_transpose

# Demo the dual projection functionality
python demo_dual_projection.py

# Test the multi-PC functionality 
python test_multi_pc_experiment.py
```

## Output
- `../prediction_steering_results/experiment_results.json`: Complete results including:
  - **Dual projection results**: All analyses with both `lm_head` and `embedding_transpose` methods
  - Original experiment: PC_1, PC_2, PC_3, PC_4, PC_5, PC_10, PC_50 sections + Centroid for each projection method
  - **Layer 31 dual analysis**: Standard vs extracted embedding comparison (when layer 31 included)
  - **Test case results**: category_word, spelling, translation analysis for each target/layer/projection
  - **Centroid perturbation results** for all analyses
  - Test cases used for each target word
  - Cross-projection comparison data
- `../prediction_steering_results/pc_similarity_matrix_PC1.png`: PC1 cosine similarity heatmap (2 decimal places)
- `../prediction_steering_results/pc_similarity_matrix_PC2.png`: PC2 cosine similarity heatmap (2 decimal places)
- `../prediction_steering_results/pc_similarity_matrix_PC3.png`: PC3 cosine similarity heatmap (2 decimal places)
- `../prediction_steering_results/pc_similarity_matrix_PC4.png`: PC4 cosine similarity heatmap (2 decimal places)
- `../prediction_steering_results/pc_similarity_matrix_PC5.png`: PC5 cosine similarity heatmap (2 decimal places)
- `../prediction_steering_results/pc_similarity_matrix_PC10.png`: PC10 cosine similarity heatmap (2 decimal places)
- `../prediction_steering_results/pc_similarity_matrix_PC50.png`: PC50 cosine similarity heatmap (2 decimal places)

## Key Changes
- **Dual Projection Methods**: Tests both LM head and embedding transpose for all analyses
- **Multi-PC Analysis**: Now tests perturbations along 1st, 2nd, 3rd, 4th, 5th, 10th, and 50th principal components
- **NEW: Centroid Perturbations**: Uses manifold centroid direction with PC1 eigenvalue magnitude
- **NEW: Layer 31 Dual Analysis**: Compares standard vs extracted embedding analysis for final layer
- **Extended PCA**: Computes up to 100 components instead of just 10
- **Confidence Filtering**: Only includes sentences where target word prediction confidence > 0.15 (15%)
- **Additional Test Cases**: Tests 3 new input types using the same PCs from the original experiment
- **Cross-Prompt Analysis**: Compare how PC perturbations affect different prompt formats
- **Cross-Projection Analysis**: Compare how projection method affects the same perturbations
- **Enhanced Visualization**: Cosine similarity matrices display values with 2 decimal places
- **Detailed Results**: Results structure includes separate sections for each projection method, PC tested, centroid, and test case type
- **Multiple Similarity Matrices**: Creates separate cosine similarity matrices for each PC

## Requirements
- PyTorch
- Transformers 
- scikit-learn
- matplotlib
- seaborn
- numpy 