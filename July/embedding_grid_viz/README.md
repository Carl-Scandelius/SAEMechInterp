# Embedding Grid Visualization

A novel approach to visualizing high-dimensional language model embedding spaces by creating 2D grid-based token maps.

## Overview

This module creates intuitive visualizations of how language models map embedding vectors to next-token predictions by:

1. **Creating a 2D Grid**: Generate a regular grid in 2D space (e.g., 100×100) centered at the origin
2. **Embedding Coordinates**: Map 2D grid coordinates to the full embedding dimensionality (e.g., 4096 for Llama-3.1-8B)
3. **Token Prediction**: For each grid point, compute the highest-probability next token
4. **Visualization**: Create a map showing predicted tokens in each grid cell

## Key Features

- **Multiple Embedding Methods**: Different ways to map 2D coordinates to high-dimensional space
- **Flexible Grid Sizing**: Configurable grid resolution and spacing
- **Memory Efficient**: Batch processing for large grids
- **High-Quality Visualization**: Professional matplotlib-based token maps
- **Comprehensive Output**: Both visual and numerical results

## Quick Start

```bash
conda activate interp

# Create a basic 50×50 grid visualization of layer 31
python core_viz.py --grid-size 50 --layer-idx 31

# Higher resolution grid with custom spacing
python core_viz.py --grid-size 100 --spacing 0.25 --embedding-method random_projection

# Smaller grid for testing
python core_viz.py --grid-size 20 --spacing 1.0 --batch-size 50
```

## Embedding Methods

### 1. **Direct** (`--embedding-method direct`)
- Places x,y coordinates in first two dimensions, zeros elsewhere
- Sparse representation focusing on spatial relationships
- Best for understanding pure coordinate effects

### 2. **Replicated** (`--embedding-method replicated`)
- Replicates x,y pattern across all dimension pairs
- Dense representation with repeated structure
- Explores redundancy in embedding space

### 3. **Random Projection** (`--embedding-method random_projection`) [Default]
- Projects 2D coordinates through random matrix to full dimensionality
- Mathematically principled approach preserving distances
- Most realistic for exploring embedding space structure

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--grid-size` | 50 | Grid dimensions (N×N points) |
| `--spacing` | 0.5 | Distance between grid points |
| `--layer-idx` | 31 | Model layer to visualize |
| `--embedding-method` | random_projection | How to embed 2D → high-D |
| `--batch-size` | 100 | Batch size for token prediction |
| `--output-dir` | embedding_grid_results | Output directory |

## Output Files

### Visualization
- `embedding_grid_layer_31_size_50_method_random_projection.png`
- High-resolution grid map with tokens and probability colors
- Color-coded cells showing prediction confidence
- Professional layout with colorbar and labels

### Numerical Results
- `embedding_grid_results_layer_31.json`
- Complete coordinate, token ID, and probability data
- Token strings for all grid points
- Configuration metadata

## Understanding the Visualization

- **Grid Cells**: Each cell represents one embedding vector
- **Token Text**: The predicted next token for that embedding
- **Cell Color**: Prediction probability (darker = higher confidence)
- **Spatial Patterns**: How token predictions vary across embedding space

## Example Use Cases

### Research Applications
- **Embedding Space Topology**: Understand how token predictions cluster
- **Model Behavior Analysis**: Visualize decision boundaries
- **Layer Comparison**: Compare embedding spaces across layers
- **Ablation Studies**: See how interventions affect prediction maps

### Educational Applications
- **LLM Intuition**: Build understanding of embedding spaces
- **Visualization Teaching**: Show how high-D spaces can be explored
- **Interactive Exploration**: Generate grids for different parameters

## Technical Details

### Memory Requirements
- **50×50 grid**: ~2GB GPU memory
- **100×100 grid**: ~8GB GPU memory
- **200×200 grid**: ~30GB GPU memory (use smaller batches)

### Computational Complexity
- Time: O(N²) for N×N grid
- Space: O(N² × embedding_dim) for activation storage
- Batching reduces peak memory usage

### Embedding Space Considerations
- Real embeddings have specific magnitude distributions
- Grid coordinates may be outside typical embedding ranges
- Different embedding methods explore different aspects of the space

## Advanced Usage

### Custom Grid Centers
```python
from core_viz import EmbeddingGridVisualizer, create_embedding_grid_map

# Create grid centered at specific coordinates
create_embedding_grid_map(
    grid_size=30,
    spacing=0.1,
    embedding_method="direct",
    output_dir="custom_grid"
)
```

### Programmatic Access
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core_viz import EmbeddingGridVisualizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Create visualizer
viz = EmbeddingGridVisualizer(model, tokenizer, layer_idx=15)

# Create custom grid
x_coords, y_coords = viz.create_2d_grid(grid_size=25, spacing=0.2)
embeddings = viz.embed_2d_coordinates(x_coords, y_coords, "random_projection")
token_ids, probs = viz.predict_tokens_for_grid(embeddings)
```

## Troubleshooting

**CUDA Out of Memory**: Reduce `--grid-size` or `--batch-size`
**Slow Processing**: Use smaller grids for initial exploration
**Unclear Visualization**: Adjust spacing or try different embedding methods
**Token Decoding Errors**: Some tokens may display as IDs rather than text

## Scientific Applications

This visualization technique enables:

1. **Embedding Space Exploration**: See how 2D slices of high-dimensional spaces map to tokens
2. **Model Interpretability**: Understand regional behavior in embedding space
3. **Comparative Analysis**: Compare different models, layers, or intervention effects
4. **Hypothesis Generation**: Identify interesting patterns for further investigation

The grid approach provides a systematic way to sample and visualize the vast embedding spaces used by modern language models, offering insights into their internal representations and decision processes. 