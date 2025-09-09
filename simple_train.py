#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Training Script - ASCII Only
Minimal training script to avoid encoding issues
"""

import sys
import os

# Fix multiprocessing on Windows
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function with minimal output"""
    try:
        from auto_retrain import AutoRetrainer
        
        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--model', type=str, default='n')
        parser.add_argument('--auto', action='store_true')
        args = parser.parse_args()
        
        print("Starting training...")
        
        # Create retrainer
        retrainer = AutoRetrainer()
        
        # Get stats
        stats = retrainer.get_training_stats()
        print(f"Feedback files: {stats['feedback_files']}")
        print(f"Total samples: {stats['total_samples']}")
        
        if stats['feedback_files'] == 0:
            print("ERROR: No feedback data found!")
            return 1
        
        # Start training
        print(f"Training with epochs={args.epochs}, model={args.model}")
        success = retrainer.start_yolo_training(epochs=args.epochs, model_size=args.model)
        
        if success:
            print("SUCCESS: Training completed!")
            return 0
        else:
            print("ERROR: Training failed!")
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
