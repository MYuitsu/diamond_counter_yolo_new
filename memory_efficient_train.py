#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Efficient Training Script
Optimized for limited GPU memory and Windows multiprocessing
"""

import sys
import os
import multiprocessing

# Windows multiprocessing fix
if __name__ == '__main__':
    multiprocessing.freeze_support()

def main():
    """Main function with memory-efficient training"""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from auto_retrain import AutoRetrainer
        
        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=20)  # Giảm epochs mặc định
        parser.add_argument('--model', type=str, default='n')
        parser.add_argument('--auto', action='store_true')
        args = parser.parse_args()
        
        print("Starting memory-efficient training...")
        print(f"Epochs: {args.epochs}, Model: {args.model}")
        
        # Create retrainer
        retrainer = AutoRetrainer()
        
        # Get stats
        stats = retrainer.get_training_stats()
        print(f"Feedback files: {stats['feedback_files']}")
        print(f"Total samples: {stats['total_samples']}")
        
        if stats['feedback_files'] == 0:
            print("ERROR: No feedback data found!")
            return 1
        
        # Start training with memory-efficient settings
        print("Starting CPU-only training to avoid memory issues...")
        success = retrainer.start_yolo_training(epochs=args.epochs, model_size=args.model)
        
        if success:
            print("SUCCESS: Training completed!")
            return 0
        else:
            print("ERROR: Training failed!")
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows fix
    exit_code = main()
    sys.exit(exit_code)
