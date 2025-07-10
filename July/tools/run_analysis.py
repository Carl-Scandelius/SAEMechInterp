#!/usr/bin/env python3
"""SAE mechanism interpretability analysis runner."""

import argparse
import sys
from typing import Optional

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for analysis configuration."""
    parser = argparse.ArgumentParser(description="Run token analysis with configurable parameters")
    parser.add_argument(
        "--script", 
        type=str, 
        choices=["word_token", "last_token"], 
        required=True,
        help="Analysis script to run"
    )
    parser.add_argument(
        "--use_system_prompt", 
        action="store_true",
        help="Use system prompt for manifold analysis"
    )
    parser.add_argument(
        "--perturb_once", 
        action="store_true",
        help="Apply perturbation only to final token of user prompt (last_token only)"
    )
    parser.add_argument(
        "--cross_concept_only", 
        action="store_true",
        help="Skip standard perturbation experiments and only run cross-concept perturbations (last_token only)"
    )
    parser.add_argument(
        "--use_pranav_sentences",
        action="store_true", 
        help="Use manifold_sentences_hard_exactword_1000.json instead of prompts.json (last_token only)"
    )
    parser.add_argument(
        "--local_centre",
        action="store_true",
        help="Use concept-specific centering (subtract each concept's own centroid) instead of global centering (last_token only)"
    )
    return parser.parse_args()

def main() -> None:
    """Execute the specified analysis script with configured parameters."""
    args = parse_arguments()
    
    if args.script == "word_token":
        print("Running WordToken analysis...")
        if args.perturb_once:
            print("Warning: --perturb_once only applicable for LastToken.py")
        if args.cross_concept_only:
            print("Warning: --cross_concept_only only applicable for LastToken.py")
            
        import wordToken
        wordToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD = args.use_system_prompt
        print(f"USE_SYSTEM_PROMPT_FOR_MANIFOLD: {wordToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
        
        wordToken.main()
        
    elif args.script == "last_token":
        print("Running LastToken analysis...")
        
        import lastToken
        lastToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD = args.use_system_prompt
        print(f"USE_SYSTEM_PROMPT_FOR_MANIFOLD: {lastToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
        
        original_main = lastToken.main
        
        def modified_main():
            lastToken.PERTURB_ONCE = args.perturb_once
            lastToken.CROSS_CONCEPT_ONLY = args.cross_concept_only
            lastToken.USE_PRANAV_SENTENCES = args.use_pranav_sentences
            lastToken.LOCAL_CENTRE = args.local_centre
            print(f"PERTURB_ONCE: {lastToken.PERTURB_ONCE}")
            print(f"CROSS_CONCEPT_ONLY: {lastToken.CROSS_CONCEPT_ONLY}")
            print(f"USE_PRANAV_SENTENCES: {lastToken.USE_PRANAV_SENTENCES}")
            print(f"LOCAL_CENTRE: {lastToken.LOCAL_CENTRE}")
            original_main()
            
        lastToken.main = modified_main
        lastToken.main()
        
    else:
        print(f"Error: Unknown script '{args.script}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
