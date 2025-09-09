# -*- coding: utf-8 -*-
"""
Safe Training Wrapper
Wrapper script to handle encoding issues when running training from Flask
"""

import sys
import os
import subprocess
import io

def run_training_safe():
    """Run training with proper encoding handling"""
    try:
        # Get command line arguments (passed from Flask)
        args = sys.argv[1:]  # Skip script name
        
        # Prepare command
        cmd = ['python', 'auto_retrain.py'] + args
        
        print(f"Starting training with args: {args}")
        
        # Run with proper encoding and redirect stdout/stderr
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Print output safely
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Training completed with return code: {result.returncode}")
        return result.returncode
        
    except Exception as e:
        print(f"Training wrapper error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_training_safe()
    sys.exit(exit_code)
