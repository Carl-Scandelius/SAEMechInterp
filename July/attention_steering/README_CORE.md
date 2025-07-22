# Core Attention Steering Experiment

Minimal, rigorous implementation of attention steering experiment for scientific research.

## Experiment Protocol

1. **Extract** final token pre-residual embeddings from Anthropic dataset on Llama 3.1-8B-Instruct
2. **Compute** PCA for each layer [0, 15, 31], keeping top 3 components by explained variance  
3. **Steer** model output by perturbing final token in direction of largest and second largest PC
4. **Generate** cosine similarity matrix of all retained PCs across layers

## Usage

```bash
pip install -r requirements_minimal.txt
python core_experiment.py
```

## Scientific Rigor

- **Deterministic generation** (do_sample=False) for reproducibility
- **Exactly 400 training samples** for PCA computation
- **Perturbations scaled by eigenvalue** for principled magnitudes
- **Complete numerical results** saved to experiment_results.json

## Outputs

- `pc_similarity_matrix.png` - Cosine similarity heatmap of all PCs
- `experiment_results.json` - Complete numerical results
- Console output with steering examples

## Implementation Details

- Uses `model.layers.{layer_idx}.self_attn` hook for pre-residual extraction
- Applies perturbation only once per generation (apply_once=True)
- Tests multiples [0.5, 1.0, 2.0] of eigenvalue for steering strength
- Computes exact cosine similarity: normalized_pc_i · normalized_pc_j 