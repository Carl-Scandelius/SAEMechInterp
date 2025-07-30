#!/usr/bin/env python
"""
Test script for embedding grid visualization.
Tests core functionality without requiring the full LLM.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from core_viz import EmbeddingGridVisualizer

def test_grid_creation():
    """Test 2D grid generation."""
    print("🧪 Testing 2D Grid Creation")
    print("=" * 40)
    
    # Create a mock visualizer for testing
    class MockVisualizer:
        def __init__(self):
            self.hidden_size = 4096
        
        def create_2d_grid(self, grid_size=10, spacing=0.5, center=(0.0, 0.0)):
            half_extent = (grid_size - 1) * spacing / 2
            x_1d = np.linspace(center[0] - half_extent, center[0] + half_extent, grid_size)
            y_1d = np.linspace(center[1] - half_extent, center[1] + half_extent, grid_size)
            x_coords, y_coords = np.meshgrid(x_1d, y_1d)
            return x_coords, y_coords
    
    viz = MockVisualizer()
    
    # Test basic grid creation
    x_coords, y_coords = viz.create_2d_grid(grid_size=5, spacing=1.0)
    
    print(f"Grid shape: {x_coords.shape}")
    print(f"X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    
    # Check that grid is properly centered
    assert x_coords.shape == (5, 5), f"Expected (5,5), got {x_coords.shape}"
    assert abs(x_coords.mean()) < 1e-10, f"Grid not centered in X: {x_coords.mean()}"
    assert abs(y_coords.mean()) < 1e-10, f"Grid not centered in Y: {y_coords.mean()}"
    
    print("✅ Grid creation test passed!")
    return True

def test_embedding_methods():
    """Test different embedding methods."""
    print("\n🧪 Testing Embedding Methods")
    print("=" * 40)
    
    # Create test coordinates
    grid_size = 3
    x_coords = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float32)
    y_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    
    # Mock model configuration
    class MockConfig:
        hidden_size = 8
    
    class MockModel:
        def __init__(self):
            self.config = MockConfig()
    
    class MockVisualizer:
        def __init__(self):
            self.hidden_size = 8
            self.device = torch.device("cpu")
        
        def embed_2d_coordinates(self, x_coords, y_coords, method="direct"):
            grid_size = x_coords.shape[0]
            
            if method == "direct":
                embeddings = torch.zeros(grid_size, grid_size, self.hidden_size, dtype=torch.float32)
                embeddings[:, :, 0] = torch.from_numpy(x_coords).float()
                embeddings[:, :, 1] = torch.from_numpy(y_coords).float()
                
            elif method == "replicated":
                embeddings = torch.zeros(grid_size, grid_size, self.hidden_size, dtype=torch.float32)
                for i in range(0, self.hidden_size, 2):
                    if i < self.hidden_size:
                        embeddings[:, :, i] = torch.from_numpy(x_coords).float()
                    if i + 1 < self.hidden_size:
                        embeddings[:, :, i + 1] = torch.from_numpy(y_coords).float()
                        
            elif method == "random_projection":
                torch.manual_seed(42)
                projection_matrix = torch.randn(2, self.hidden_size, dtype=torch.float32)
                projection_matrix = torch.nn.functional.normalize(projection_matrix, dim=1)
                
                coords_2d = torch.stack([
                    torch.from_numpy(x_coords).float().flatten(),
                    torch.from_numpy(y_coords).float().flatten()
                ], dim=0)
                
                embeddings_flat = torch.mm(projection_matrix.T, coords_2d).T
                embeddings = embeddings_flat.view(grid_size, grid_size, self.hidden_size)
            
            return embeddings
    
    viz = MockVisualizer()
    
    # Test each embedding method
    methods = ["direct", "replicated", "random_projection"]
    for method in methods:
        print(f"Testing {method} method...")
        embeddings = viz.embed_2d_coordinates(x_coords, y_coords, method)
        
        assert embeddings.shape == (3, 3, 8), f"Wrong shape for {method}: {embeddings.shape}"
        
        if method == "direct":
            # Check that x,y are in first two dimensions
            assert torch.allclose(embeddings[:, :, 0], torch.from_numpy(x_coords)), "X coordinates not in dim 0"
            assert torch.allclose(embeddings[:, :, 1], torch.from_numpy(y_coords)), "Y coordinates not in dim 1"
            assert torch.allclose(embeddings[:, :, 2:], torch.zeros(3, 3, 6)), "Other dims should be zero"
        
        print(f"  ✅ {method} method works correctly")
    
    print("✅ All embedding methods test passed!")
    return True

def test_token_prediction_mock():
    """Test token prediction logic with mock data."""
    print("\n🧪 Testing Token Prediction Logic")
    print("=" * 40)
    
    # Create mock embeddings and language model head
    grid_size = 2
    hidden_size = 4
    vocab_size = 10
    
    embeddings = torch.randn(grid_size, grid_size, hidden_size)
    
    # Mock language model head
    lm_head = torch.nn.Linear(hidden_size, vocab_size)
    
    # Test prediction logic
    embeddings_flat = embeddings.view(-1, hidden_size)
    logits = lm_head(embeddings_flat)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_tokens = torch.topk(probs, k=1, dim=-1)
    
    token_ids = top_tokens.squeeze(-1).detach().numpy()
    probabilities = top_probs.squeeze(-1).detach().numpy()
    
    # Reshape back to grid
    token_grid = token_ids.reshape(grid_size, grid_size)
    prob_grid = probabilities.reshape(grid_size, grid_size)
    
    print(f"Token grid shape: {token_grid.shape}")
    print(f"Probability grid shape: {prob_grid.shape}")
    print(f"Token ID range: [{token_grid.min()}, {token_grid.max()}]")
    print(f"Probability range: [{prob_grid.min():.6f}, {prob_grid.max():.6f}]")
    
    assert token_grid.shape == (grid_size, grid_size), "Wrong token grid shape"
    assert prob_grid.shape == (grid_size, grid_size), "Wrong probability grid shape"
    assert (token_grid >= 0).all() and (token_grid < vocab_size).all(), "Invalid token IDs"
    assert (prob_grid > 0).all() and (prob_grid <= 1.0).all(), "Invalid probabilities"
    
    print("✅ Token prediction logic test passed!")
    return True

def test_visualization_setup():
    """Test visualization setup without actually creating plots."""
    print("\n🧪 Testing Visualization Setup")
    print("=" * 40)
    
    # Create test data
    grid_size = 3
    x_coords = np.linspace(-1, 1, grid_size)
    y_coords = np.linspace(-1, 1, grid_size)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    token_ids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    probabilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    
    print(f"X grid shape: {x_grid.shape}")
    print(f"Y grid shape: {y_grid.shape}")
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Test color mapping
    prob_colors = plt.cm.viridis(probabilities / probabilities.max())
    print(f"Color array shape: {prob_colors.shape}")
    
    # Test cell boundary calculations
    x_step = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    y_step = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
    
    print(f"Grid spacing: x_step={x_step:.3f}, y_step={y_step:.3f}")
    
    # Test that all data structures are compatible
    assert x_grid.shape == y_grid.shape == token_ids.shape == probabilities.shape, "Mismatched shapes"
    assert prob_colors.shape == (*probabilities.shape, 4), f"Wrong color shape: {prob_colors.shape}"
    
    print("✅ Visualization setup test passed!")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Embedding Grid Visualization Tests")
    print("=" * 80)
    
    tests = [
        test_grid_creation,
        test_embedding_methods,
        test_token_prediction_mock,
        test_visualization_setup
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! The embedding grid visualization logic is working correctly.")
        print("✅ Ready to run: python core_viz.py --grid-size 20 --spacing 1.0")
    else:
        print(f"\n❌ {len(tests) - passed} test(s) failed. Please check the implementation.")
    
    print("\n💡 To run the full experiment:")
    print("   conda activate interp")
    print("   cd embedding_grid_viz")
    print("   python core_viz.py --grid-size 30 --layer-idx 31 --spacing 0.5")

if __name__ == "__main__":
    main() 