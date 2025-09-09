#!/usr/bin/env python3
"""Quick debug script"""

import sys
import os

try:
    from auto_retrain import AutoRetrainer
    
    print("Creating AutoRetrainer...")
    retrainer = AutoRetrainer()
    
    print("Getting stats...")
    stats = retrainer.get_training_stats()
    print(f"Stats: {stats}")
    
    if stats['feedback_files'] > 0:
        print("Testing process_feedback_to_roboflow...")
        success = retrainer.process_feedback_to_roboflow("detect")
        print(f"Processing success: {success}")
        
        if success:
            print("Testing training...")
            result = retrainer.start_yolo_training(epochs=2, model_size='n')
            print(f"Training result: {result}")
    else:
        print("No feedback files found!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
