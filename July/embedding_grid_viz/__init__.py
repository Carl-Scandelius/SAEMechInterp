"""
Embedding Grid Visualization

Creates a 2D grid-based visualization of language model embedding spaces by:
1. Creating a regular grid in 2D space
2. Mapping grid coordinates to high-dimensional embedding vectors
3. Computing next-token predictions for each grid point
4. Visualizing the resulting token map

This provides an intuitive way to explore how the model's embedding space
relates to token predictions in different regions.
"""

from .core_viz import create_embedding_grid_map, EmbeddingGridVisualizer

__version__ = "1.0.0"
__all__ = ["create_embedding_grid_map", "EmbeddingGridVisualizer"] 