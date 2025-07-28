# Fisher-guided Embedding Perturbations Experiment

Rigorous implementation of Fisher Information Matrix-guided perturbation analysis for language model interpretation.

## Experiment Protocol

1. **Data Selection**: Extract 20 diverse sentences containing "animals" from manifold dataset
2. **Layer Analysis**: Iterate over layers ℓ ∈ {0, 5, 10, 15, 20, 25, 30, 31}
3. **Per-sentence, per-layer procedure**:
   - Extract residual stream activation h at token immediately before "animals"
   - Compute local Fisher Information Matrix: I(h) = Σ_c p(c|h)[∇_h log p(c|h) ∇_h log p(c|h)^T]
   - Eigendecompose I(h) to find top eigenvector v with eigenvalue λ
   - Perturb with scales α ∈ {0, ±0.5, ±1, ±2, ±5, ±10} × λ
   - Record next-token predictions for h' = h + αv

## Mathematical Foundation

The Fisher Information Matrix captures the local curvature of the log-likelihood surface:
- I(h) quantifies sensitivity of predictions to perturbations around h
- Top eigenvector v indicates direction of maximal information
- Perturbations along v test model's robustness in information-rich directions

## Usage

```bash
conda activate interp
pip install -r requirements_minimal.txt
python core_experiment.py
```

## Scientific Rigor

- **Deterministic generation** for reproducibility
- **20 diverse sentences** for statistical reliability
- **Multiple perturbation scales** for comprehensive analysis
- **Fisher-guided directions** for principled perturbations

## Outputs

- `fisher_similarity_matrix.png` - Cosine similarity heatmap of Fisher eigenvectors
- `perturbation_analysis.png` - Top-5 prediction changes across scales
- `experiment_results.json` - Complete numerical results
- Console output with detailed per-sentence analysis

## Implementation Details

- Uses residual stream activations (model.layers.{layer_idx} output)
- Computes exact Fisher matrix via autograd
- Efficient eigendecomposition with torch.linalg.eigh
- Handles both GPU and CPU execution gracefully 