#!/usr/bin/env python
"""
Core implementation for embedding grid visualization.

This module creates 2D maps of high-dimensional embedding spaces by:
1. Creating a regular grid in 2D space
2. Embedding grid coordinates into the full embedding dimensionality
3. Computing next-token predictions for each grid point
4. Visualizing the results as a token map
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA

# Robust CUDA setup and device detection
def setup_device():
    """Set up the computation device with robust CUDA handling."""
    
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            test_tensor = torch.tensor([1.0]).cuda()
            device = torch.device("cuda")
            print(f"✅ Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            print(f"⚠️  CUDA available but failed initialization: {e}")
            print("   Falling back to CPU")
            return torch.device("cpu")
    else:
        print("ℹ️  CUDA not available, using CPU")
        return torch.device("cpu")

# Set up device early
DEVICE = setup_device()

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class EmbeddingGridVisualizer:
    """Create 2D grid-based visualizations of embedding spaces."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, layer_idx: int = 31):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = next(model.parameters()).device
        self.hidden_size = model.config.hidden_size
        
        print(f"Initialized visualizer for layer {layer_idx}")
        print(f"Hidden size: {self.hidden_size}")
        
    def create_2d_grid(self, grid_size: int = 100, spacing: float = 0.5, 
                      center: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a 2D grid centered at the origin.
        
        Args:
            grid_size: Number of grid points along each axis
            spacing: Distance between grid lines
            center: Center point of the grid
            
        Returns:
            x_coords, y_coords: 2D arrays of grid coordinates
        """
        # Create 1D coordinate arrays
        half_extent = (grid_size - 1) * spacing / 2
        x_1d = np.linspace(center[0] - half_extent, center[0] + half_extent, grid_size)
        y_1d = np.linspace(center[1] - half_extent, center[1] + half_extent, grid_size)
        
        # Create 2D grid
        x_coords, y_coords = np.meshgrid(x_1d, y_1d)
        
        print(f"Created {grid_size}x{grid_size} grid with spacing {spacing}")
        print(f"Grid extent: x=[{x_1d[0]:.2f}, {x_1d[-1]:.2f}], y=[{y_1d[0]:.2f}, {y_1d[-1]:.2f}]")
        
        return x_coords, y_coords
    
    def embed_2d_coordinates(self, x_coords: np.ndarray, y_coords: np.ndarray, 
                           embedding_method: str = "direct") -> torch.Tensor:
        """
        Map 2D grid coordinates to high-dimensional embedding vectors.
        
        Args:
            x_coords, y_coords: 2D coordinate arrays
            embedding_method: How to embed 2D coords into high-D space
                - "direct": Put x,y in first two dimensions, zero elsewhere
                - "replicated": Replicate x,y pattern across dimensions
                - "random_projection": Random linear combination
                - "pca_basis": Use PCA basis from real embeddings
                
        Returns:
            embeddings: Tensor of shape [grid_size, grid_size, hidden_size]
        """
        grid_size = x_coords.shape[0]
        
        if embedding_method == "direct":
            # Put x,y coordinates in first two dimensions
            embeddings = torch.zeros(grid_size, grid_size, self.hidden_size, 
                                   device=self.device, dtype=torch.float32)
            embeddings[:, :, 0] = torch.from_numpy(x_coords).float()
            embeddings[:, :, 1] = torch.from_numpy(y_coords).float()
            
        elif embedding_method == "replicated":
            # Replicate the x,y pattern across all dimensions in pairs
            embeddings = torch.zeros(grid_size, grid_size, self.hidden_size, 
                                   device=self.device, dtype=torch.float32)
            for i in range(0, self.hidden_size, 2):
                if i < self.hidden_size:
                    embeddings[:, :, i] = torch.from_numpy(x_coords).float()
                if i + 1 < self.hidden_size:
                    embeddings[:, :, i + 1] = torch.from_numpy(y_coords).float()
                    
        elif embedding_method == "random_projection":
            # Create random projection matrix
            torch.manual_seed(42)  # For reproducibility
            projection_matrix = torch.randn(2, self.hidden_size, device=self.device, dtype=torch.float32)
            projection_matrix = F.normalize(projection_matrix, dim=1)
            
            # Project 2D coordinates to high-D space
            coords_2d = torch.stack([
                torch.from_numpy(x_coords).float().flatten(),
                torch.from_numpy(y_coords).float().flatten()
            ], dim=0).to(self.device)  # [2, grid_size^2]
            
            embeddings_flat = torch.mm(projection_matrix.T, coords_2d).T  # [grid_size^2, hidden_size]
            embeddings = embeddings_flat.view(grid_size, grid_size, self.hidden_size)
            
        elif embedding_method == "pca_basis":
            # This would require real embeddings to compute PCA basis
            # For now, fall back to random projection
            print("PCA basis method requires real embeddings - using random projection instead")
            return self.embed_2d_coordinates(x_coords, y_coords, "random_projection")
            
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
        
        print(f"Embedded 2D coordinates using '{embedding_method}' method")
        print(f"Embedding tensor shape: {embeddings.shape}")
        print(f"Embedding value range: [{embeddings.min().item():.3f}, {embeddings.max().item():.3f}]")
        
        return embeddings
    
    def predict_tokens_for_grid(self, embeddings: torch.Tensor, 
                              batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute next-token predictions for each point in the embedding grid.
        
        Args:
            embeddings: Grid of embedding vectors [grid_size, grid_size, hidden_size]
            batch_size: Number of embeddings to process at once
            
        Returns:
            token_ids: Grid of most likely token IDs [grid_size, grid_size]
            probabilities: Grid of corresponding probabilities [grid_size, grid_size]
        """
        grid_size = embeddings.shape[0]
        token_ids = np.zeros((grid_size, grid_size), dtype=int)
        probabilities = np.zeros((grid_size, grid_size), dtype=float)
        
        # Flatten embeddings for batch processing
        embeddings_flat = embeddings.view(-1, self.hidden_size)  # [grid_size^2, hidden_size]
        total_points = embeddings_flat.shape[0]
        
        print(f"Computing predictions for {total_points} grid points...")
        
        # Process in batches to manage memory
        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            batch_embeddings = embeddings_flat[start_idx:end_idx]
            
            if start_idx % (batch_size * 10) == 0:
                print(f"  Processing batch {start_idx // batch_size + 1}/{(total_points + batch_size - 1) // batch_size}")
            
            with torch.no_grad():
                # Ensure embeddings are in the right dtype for the language modeling head
                if batch_embeddings.dtype != self.model.lm_head.weight.dtype:
                    batch_embeddings = batch_embeddings.to(dtype=self.model.lm_head.weight.dtype)
                
                # Get logits from language modeling head
                logits = self.model.lm_head(batch_embeddings)  # [batch_size, vocab_size]
                probs = F.softmax(logits, dim=-1)
                
                # Get most likely tokens
                top_probs, top_tokens = torch.topk(probs, k=1, dim=-1)
                
                # Store results
                batch_token_ids = top_tokens.squeeze(-1).cpu().numpy()
                batch_probs = top_probs.squeeze(-1).cpu().numpy()
                
                # Map back to grid coordinates
                for i, (token_id, prob) in enumerate(zip(batch_token_ids, batch_probs)):
                    global_idx = start_idx + i
                    row = global_idx // grid_size
                    col = global_idx % grid_size
                    token_ids[row, col] = token_id
                    probabilities[row, col] = prob
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available() and start_idx % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        print(f"✅ Computed predictions for all {total_points} grid points")
        print(f"Probability range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
        
        return token_ids, probabilities
    
    def create_token_visualization(self, x_coords: np.ndarray, y_coords: np.ndarray,
                                 token_ids: np.ndarray, probabilities: np.ndarray,
                                 output_path: str, figsize: Tuple[int, int] = (20, 20),
                                 max_display_size: int = 50) -> None:
        """
        Create a visualization of the token grid map.
        
        Args:
            x_coords, y_coords: Grid coordinate arrays
            token_ids: Token ID predictions for each grid point
            probabilities: Prediction probabilities
            output_path: Where to save the visualization
            figsize: Figure size in inches
            max_display_size: Maximum grid size to display (for readability)
        """
        grid_size = token_ids.shape[0]
        
        # Downsample if grid is too large for readable visualization
        if grid_size > max_display_size:
            step = grid_size // max_display_size
            x_coords = x_coords[::step, ::step]
            y_coords = y_coords[::step, ::step]
            token_ids = token_ids[::step, ::step]
            probabilities = probabilities[::step, ::step]
            display_size = x_coords.shape[0]
            print(f"Downsampled {grid_size}x{grid_size} grid to {display_size}x{display_size} for visualization")
        else:
            display_size = grid_size
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate cell size
        x_step = x_coords[0, 1] - x_coords[0, 0] if display_size > 1 else 1.0
        y_step = y_coords[1, 0] - y_coords[0, 0] if display_size > 1 else 1.0
        
        # Create color map based on probabilities
        prob_colors = plt.cm.viridis(probabilities / probabilities.max())
        
        print(f"Creating visualization with {display_size}x{display_size} cells...")
        
        # Draw grid and tokens
        for i in range(display_size):
            for j in range(display_size):
                # Get cell boundaries (token is shown in box with top-right at grid point)
                x_center = x_coords[i, j]
                y_center = y_coords[i, j]
                
                # Cell boundaries: box with top-right corner at (x_center, y_center)
                left = x_center - x_step
                right = x_center
                bottom = y_center - y_step
                top = y_center
                
                # Draw cell background with color based on probability
                rect = patches.Rectangle((left, bottom), x_step, y_step,
                                       facecolor=prob_colors[i, j], 
                                       edgecolor='black', linewidth=0.5, alpha=0.7)
                ax.add_patch(rect)
                
                # Decode token and add text
                token_id = token_ids[i, j]
                try:
                    token_str = self.tokenizer.decode([token_id])
                    # Clean up token string for display
                    token_str = token_str.replace('▁', ' ').strip()
                    if len(token_str) > 8:  # Truncate very long tokens
                        token_str = token_str[:8] + '...'
                except:
                    token_str = f"#{token_id}"
                
                # Add token text in center of cell
                text_x = left + x_step / 2
                text_y = bottom + y_step / 2
                
                # Choose text color based on background brightness
                brightness = np.mean(prob_colors[i, j][:3])
                text_color = 'white' if brightness < 0.5 else 'black'
                
                ax.text(text_x, text_y, token_str, 
                       ha='center', va='center', fontsize=8, 
                       color=text_color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
        
        # Set up axes
        ax.set_xlim(x_coords.min() - x_step, x_coords.max())
        ax.set_ylim(y_coords.min() - y_step, y_coords.max())
        ax.set_xlabel('X Coordinate', fontsize=14)
        ax.set_ylabel('Y Coordinate', fontsize=14)
        ax.set_title(f'Embedding Grid Token Map (Layer {self.layer_idx})\n'
                    f'Grid Size: {display_size}x{display_size}, Cell Size: {x_step:.2f}', fontsize=16)
        
        # Add colorbar for probabilities
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=probabilities.min(), vmax=probabilities.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Prediction Probability', fontsize=12)
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization to {output_path}")

def create_embedding_grid_map(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    layer_idx: int = 31,
    grid_size: int = 100,
    spacing: float = 0.5,
    embedding_method: str = "random_projection",
    output_dir: str = "embedding_grid_results",
    batch_size: int = 100,
    max_display_size: int = 50,
    figsize: Tuple[int, int] = (20, 20)
) -> Dict:
    """
    Create a complete embedding grid visualization.
    
    Args:
        model_name: Hugging Face model name
        layer_idx: Which layer to visualize (default: 31 for final layer)
        grid_size: Size of the grid (grid_size x grid_size points)
        spacing: Distance between grid points
        embedding_method: How to map 2D coordinates to embeddings
        output_dir: Directory to save results
        batch_size: Batch size for prediction computation
        max_display_size: Maximum grid size for visualization
        figsize: Figure size for visualization
        
    Returns:
        Dictionary with results including coordinates, predictions, and file paths
    """
    
    print("=" * 60)
    print("EMBEDDING GRID VISUALIZATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Grid: {grid_size}x{grid_size} with spacing {spacing}")
    print(f"Embedding method: {embedding_method}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        if DEVICE.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True
            )
            model = model.to(DEVICE)
        
        model.eval()
        print(f"   Model loaded successfully on {model.device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # 2. Initialize visualizer
    print("\n2. Initializing embedding grid visualizer...")
    visualizer = EmbeddingGridVisualizer(model, tokenizer, layer_idx)
    
    # 3. Create 2D grid
    print("\n3. Creating 2D coordinate grid...")
    x_coords, y_coords = visualizer.create_2d_grid(grid_size, spacing)
    
    # 4. Embed coordinates into high-dimensional space
    print("\n4. Embedding 2D coordinates into embedding space...")
    embeddings = visualizer.embed_2d_coordinates(x_coords, y_coords, embedding_method)
    
    # 5. Compute token predictions for grid
    print("\n5. Computing next-token predictions for grid points...")
    token_ids, probabilities = visualizer.predict_tokens_for_grid(embeddings, batch_size)
    
    # 6. Create visualization
    print("\n6. Creating token grid visualization...")
    viz_path = f"{output_dir}/embedding_grid_layer_{layer_idx}_size_{grid_size}_method_{embedding_method}.png"
    visualizer.create_token_visualization(
        x_coords, y_coords, token_ids, probabilities, 
        viz_path, figsize, max_display_size
    )
    
    # 7. Save numerical results
    print("\n7. Saving numerical results...")
    results = {
        'config': {
            'model_name': model_name,
            'layer_idx': layer_idx,
            'grid_size': grid_size,
            'spacing': spacing,
            'embedding_method': embedding_method
        },
        'coordinates': {
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist()
        },
        'predictions': {
            'token_ids': token_ids.tolist(),
            'probabilities': probabilities.tolist()
        },
        'token_map': {}
    }
    
    # Add token strings to results
    for i in range(token_ids.shape[0]):
        for j in range(token_ids.shape[1]):
            token_id = int(token_ids[i, j])
            try:
                token_str = tokenizer.decode([token_id])
                results['token_map'][f"{i},{j}"] = {
                    'token_id': token_id,
                    'token_str': token_str,
                    'probability': float(probabilities[i, j]),
                    'coordinates': [float(x_coords[i, j]), float(y_coords[i, j])]
                }
            except:
                results['token_map'][f"{i},{j}"] = {
                    'token_id': token_id,
                    'token_str': f"TOKEN_{token_id}",
                    'probability': float(probabilities[i, j]),
                    'coordinates': [float(x_coords[i, j]), float(y_coords[i, j])]
                }
    
    results_path = f"{output_dir}/embedding_grid_results_layer_{layer_idx}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Saved numerical results to {results_path}")
    print(f"\n✅ Embedding grid visualization completed successfully!")
    print(f"   Visualization: {viz_path}")
    print(f"   Results: {results_path}")
    
    return {
        'visualization_path': viz_path,
        'results_path': results_path,
        'results': results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embedding grid visualization")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B",
                       help="Model name")
    parser.add_argument("--layer-idx", type=int, default=31,
                       help="Layer index to visualize")
    parser.add_argument("--grid-size", type=int, default=50,
                       help="Grid size (grid_size x grid_size)")
    parser.add_argument("--spacing", type=float, default=0.5,
                       help="Spacing between grid points")
    parser.add_argument("--embedding-method", default="random_projection",
                       choices=["direct", "replicated", "random_projection"],
                       help="Method to embed 2D coordinates")
    parser.add_argument("--output-dir", default="embedding_grid_results",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for prediction computation")
    
    args = parser.parse_args()
    
    create_embedding_grid_map(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        grid_size=args.grid_size,
        spacing=args.spacing,
        embedding_method=args.embedding_method,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    ) 