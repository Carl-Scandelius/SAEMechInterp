#!/usr/bin/env python3
"""
Runner script for SAE Mechanism Interpretability analysis.

This script allows you to run either WordToken.py or LastToken.py with configurable parameters.

Examples:
    # Run LastToken.py with default settings
    python run_analysis.py --script last_token
    
    # Run WordToken.py with system prompt for manifold
    python run_analysis.py --script word_token --use_system_prompt
    
    # Run LastToken.py with system prompt and applying the perturbation only to final token of the user's prompt
    python run_analysis.py --script last_token --use_system_prompt --perturb_once
"""

import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run token analysis with configurable parameters")
    parser.add_argument(
        "--script", 
        type=str, 
        choices=["word_token", "last_token"], 
        required=True,
    )
    parser.add_argument(
        "--use_system_prompt", 
        action="store_true",
    )
    parser.add_argument(
        "--perturb_once", 
        action="store_true",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.script == "word_token":
        print("Running WordToken analysis...")
        if args.perturb_once:
            print("Warning: --perturb_once is only applicable for LastToken.py and will be ignored")
            
        import wordToken
        wordToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD = args.use_system_prompt
        print(f"USE_SYSTEM_PROMPT_FOR_MANIFOLD set to: {wordToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
        
        wordToken.main()
        
    elif args.script == "last_token":
        print("Running LastToken analysis...")
        
        import LastToken
        LastToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD = args.use_system_prompt
        print(f"USE_SYSTEM_PROMPT_FOR_MANIFOLD set to: {LastToken.USE_SYSTEM_PROMPT_FOR_MANIFOLD}")
        
        original_main = LastToken.main
        
        def modified_main():
            LastToken.PERTURB_ONCE = args.perturb_once
            print(f"PERTURB_ONCE set to: {LastToken.PERTURB_ONCE}")
            original_main()
            
        LastToken.main = modified_main
        LastToken.main()
        
    else:
        print(f"Error: Unknown script '{args.script}'")
        sys.exit(1)

if __name__ == "__main__":
    main()