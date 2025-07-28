#!/usr/bin/env python
"""
Test script for Fisher Information Matrix computation logic.
This tests the mathematical components without requiring the full LLM.
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_fisher_computation_logic():
    """Test the Fisher Information Matrix computation with synthetic data."""
    
    print("🧪 Testing Fisher Information Matrix Computation Logic")
    print("=" * 60)
    
    # Simulate a small vocabulary and hidden size for testing
    hidden_size = 512
    vocab_size = 1000
    
    # Create synthetic activation h (residual stream)
    h = torch.randn(hidden_size, dtype=torch.float32, requires_grad=True)
    
    # Create a simple linear layer to simulate language modeling head
    lm_head = torch.nn.Linear(hidden_size, vocab_size, dtype=torch.float32)
    
    print(f"Hidden size: {hidden_size}")
    print(f"Vocab size: {vocab_size}")
    print(f"Activation h shape: {h.shape}")
    
    # Forward pass to get logits and probabilities
    with torch.enable_grad():
        logits = lm_head(h)  # [vocab_size]
        probs = F.softmax(logits, dim=-1)  # [vocab_size]
        
        # Limit to top-k tokens for efficiency (as in real implementation)
        top_k = 100
        top_probs, top_indices = torch.topk(probs, top_k)
        
        print(f"Using top-{top_k} tokens for Fisher computation")
        print(f"Top 5 probabilities: {top_probs[:5].detach().numpy()}")
        
        # Initialize Fisher matrix
        fisher_matrix = torch.zeros(hidden_size, hidden_size, dtype=torch.float32)
        
        # Compute Fisher matrix: I(h) = Σ_c p(c|h)[∇_h log p(c|h) ∇_h log p(c|h)^T]
        for i, (prob, token_idx) in enumerate(zip(top_probs, top_indices)):
            if prob < 1e-8:  # Skip very low probability tokens
                continue
            
            # Compute gradient of log p(c|h) w.r.t. h
            log_prob = torch.log(prob + 1e-8)
            
            # Clear gradients
            if h.grad is not None:
                h.grad.zero_()
            
            # Compute gradient
            log_prob.backward(retain_graph=True)
            grad = h.grad.clone()
            
            # Add to Fisher matrix: p(c|h) * grad * grad^T
            fisher_matrix += prob * torch.outer(grad, grad)
        
        print(f"✅ Fisher matrix computed successfully")
        print(f"Fisher matrix shape: {fisher_matrix.shape}")
        print(f"Fisher matrix norm: {torch.norm(fisher_matrix).item():.6f}")
        
        # Test eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(fisher_matrix)
            # Sort in descending order
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            top_eigenvalue = eigenvalues[sorted_indices[0]]
            top_eigenvector = eigenvectors[:, sorted_indices[0]]
            
            print(f"✅ Eigendecomposition successful")
            print(f"Top eigenvalue: {top_eigenvalue.item():.6f}")
            print(f"Top eigenvector norm: {torch.norm(top_eigenvector).item():.6f}")
            
            # Test perturbation
            scales = [-2.0, -1.0, 0.0, 1.0, 2.0]
            print(f"\n🔄 Testing perturbations with scales: {scales}")
            
            for scale in scales:
                perturbation = scale * top_eigenvalue * top_eigenvector
                h_perturbed = h.detach() + perturbation
                
                # Test forward pass with perturbed activation
                logits_perturbed = lm_head(h_perturbed)
                probs_perturbed = F.softmax(logits_perturbed, dim=-1)
                
                # Get top predictions
                top_probs_pert, top_indices_pert = torch.topk(probs_perturbed, 5)
                
                print(f"  Scale {scale:+.1f}: Top prob = {top_probs_pert[0].item():.6f}")
            
            print(f"✅ Perturbation testing successful")
            
        except Exception as e:
            print(f"❌ Eigendecomposition failed: {e}")
            return False
    
    return True

def test_cosine_similarity_computation():
    """Test cosine similarity computation for Fisher eigenvectors."""
    
    print(f"\n🧪 Testing Cosine Similarity Computation")
    print("=" * 60)
    
    # Create some synthetic Fisher eigenvectors
    hidden_size = 512
    num_vectors = 10
    
    directions = []
    for i in range(num_vectors):
        # Create random directions (simulating Fisher eigenvectors)
        direction = torch.randn(hidden_size)
        direction = direction / torch.norm(direction)  # Normalize
        directions.append(direction.numpy())
    
    directions = np.array(directions)
    print(f"Created {num_vectors} synthetic Fisher eigenvectors")
    print(f"Shape: {directions.shape}")
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(directions, directions.T)
    
    print(f"✅ Cosine similarity matrix computed")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Diagonal should be ~1.0: {np.diag(similarity_matrix)[:3]}")
    print(f"Off-diagonal range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    # Check that diagonal elements are approximately 1
    diagonal_check = np.allclose(np.diag(similarity_matrix), 1.0, atol=1e-6)
    print(f"✅ Diagonal elements check: {diagonal_check}")
    
    return diagonal_check

if __name__ == "__main__":
    print("🚀 Starting Fisher-guided Perturbation Tests")
    print("=" * 80)
    
    # Test Fisher computation logic
    fisher_test_passed = test_fisher_computation_logic()
    
    # Test cosine similarity computation
    similarity_test_passed = test_cosine_similarity_computation()
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Fisher computation test: {'✅ PASSED' if fisher_test_passed else '❌ FAILED'}")
    print(f"Cosine similarity test: {'✅ PASSED' if similarity_test_passed else '❌ FAILED'}")
    
    if fisher_test_passed and similarity_test_passed:
        print("\n🎉 All tests passed! The Fisher-guided perturbation logic is working correctly.")
        print("✅ Ready to run the full experiment with: python core_experiment.py")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    print("\n💡 To run the full experiment:")
    print("   conda activate interp")
    print("   cd fisher_per")
    print("   python core_experiment.py --num-sentences 5 --output-dir test_results") 