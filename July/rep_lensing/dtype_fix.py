#!/usr/bin/env python3
"""
Patch to fix dtype mismatch issues in representation lensing experiment.
Apply these changes to the core_experiment.py file.
"""

# Main fixes needed:

# 1. In LanguageModelingAnalyzer.get_top_predictions method:
# OLD:
# embeddings = embeddings.to(self.device)
# NEW:
embeddings = embeddings.to(self.device, dtype=self.lm_head.weight.dtype)

# 2. In PerturbationAnalyzer.perturb_and_analyze method:
# OLD:
# perturbed_tensor = torch.from_numpy(final_embeddings).float()
# NEW:
perturbed_tensor = torch.from_numpy(final_embeddings).to(dtype=embeddings.dtype)

# The core issue:
# - Model is loaded with torch.float16 (half precision)
# - But embeddings get converted to float32 during numpy operations
# - This causes a dtype mismatch when passing through lm_head
# - Solution: Always preserve the model's dtype (float16) when converting back to tensors

print("Apply these fixes to resolve the dtype mismatch error!") 