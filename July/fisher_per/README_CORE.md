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

# For memory-constrained environments (A100 40GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python core_experiment.py --num-sentences 10 --max-vocab-subset 150 --batch-size 15

# Note: Float16/Float32 dtype issues are handled automatically

# Standard usage (requires more memory)
python core_experiment.py --num-sentences 20 --max-vocab-subset 200 --batch-size 20

# Ultra-low memory usage
python core_experiment.py --num-sentences 5 --max-vocab-subset 100 --batch-size 10
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

## Memory Optimization

For large models like Llama-3.1-8B on GPUs with limited memory:

- **--max-vocab-subset**: Reduces vocabulary tokens used in Fisher computation (default: 200)
- **--batch-size**: Processes Fisher gradients in smaller batches (default: 20)
- **--num-sentences**: Use fewer sentences for initial testing (recommend: 5-10)
- **Environment**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Memory Requirements**:
- A100 40GB: Use `--max-vocab-subset 150 --batch-size 15 --num-sentences 10`
- V100 32GB: Use `--max-vocab-subset 100 --batch-size 10 --num-sentences 5`
- RTX 4090 24GB: Use `--max-vocab-subset 80 --batch-size 8 --num-sentences 3`

## Common Issues & Solutions

**"linalg_eigh_cuda" not implemented for 'Half'**: 
- **Problem**: PyTorch's eigendecomposition doesn't support float16 on CUDA
- **Solution**: Automatically casts Fisher matrix computation to float32 for numerical stability
- **Implementation**: 
  - Converts activations to float32 before gradient computation
  - Converts language modeling head weights to float32 temporarily
  - Performs eigendecomposition in float32
  - Converts results back to original dtype for consistency
- **Impact**: No user action required, maintains model precision for other operations

**"Can't call numpy() on Tensor that requires grad"**: 
- **Problem**: Fisher eigenvectors retain gradients from computation, blocking numpy conversion
- **Solution**: Automatically detaches tensors from computation graph before numpy operations
- **Implementation**: Added `.detach()` calls in both Fisher matrix computation and cosine similarity calculation
- **Impact**: No user action required, preserves computational efficiency 