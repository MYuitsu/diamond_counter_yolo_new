#!/usr/bin/env python3
"""
Test Transfer Learning Workflow
==============================

Script để test toàn bộ quá trình Active Learning với Transfer Learning:
1. Tạo fake feedback data
2. Chạy transfer learning
3. Kiểm tra model mới
4. So sánh performance

Usage:
    python test_transfer_learning.py
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
import shutil
from auto_retrain import AutoRetrainer
from ultralytics import YOLO

def create_fake_feedback_data():
    """Tạo fake feedback data để test"""
    print("📋 Creating fake feedback data for testing...")
    
    # Tạo thư mục feedback
    feedback_dir = "active_learning_data"
    os.makedirs(f"{feedback_dir}/images", exist_ok=True)
    os.makedirs(f"{feedback_dir}/annotations", exist_ok=True)
    
    # Tạo 5 fake feedback files
    for i in range(5):
        timestamp = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
        
        # Tạo fake image (640x640 black image)
        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(fake_image, f"Test Image {i+1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        image_path = f"{feedback_dir}/images/feedback_{timestamp}.jpg"
        cv2.imwrite(image_path, fake_image)
        
        # Tạo fake annotation với true positives và missed objects
        annotation_data = {
            'image_path': image_path,
            'timestamp': timestamp,
            'predictions': [
                {'x': 100 + i*50, 'y': 100, 'w': 50, 'h': 50, 'confidence': 0.8 + i*0.05},
                {'x': 200 + i*30, 'y': 150, 'w': 45, 'h': 45, 'confidence': 0.75 + i*0.03}
            ],
            'user_corrections': {
                'false_positives': [1] if i % 2 == 0 else [],  # Một số có false positive
                'missed_objects': [
                    {'x': 300 + i*20, 'y': 200, 'w': 40, 'h': 40}
                ] if i % 3 == 0 else [],  # Một số có missed objects
                'true_positives': [
                    {'x': 100 + i*50, 'y': 100, 'w': 50, 'h': 50, 'confidence': 0.8 + i*0.05}
                ] if i % 2 == 0 else [  # Nếu có false positive thì chỉ có 1 true positive
                    {'x': 100 + i*50, 'y': 100, 'w': 50, 'h': 50, 'confidence': 0.8 + i*0.05},
                    {'x': 200 + i*30, 'y': 150, 'w': 45, 'h': 45, 'confidence': 0.75 + i*0.03}
                ]  # Nếu không có false positive thì có 2 true positives
            },
            'annotation_stats': {
                'true_positives': 1 if i % 2 == 0 else 2,
                'false_positives': 1 if i % 2 == 0 else 0,
                'missed_objects': 1 if i % 3 == 0 else 0,
                'total_predictions': 2,
                'total_ground_truth': (1 if i % 2 == 0 else 2) + (1 if i % 3 == 0 else 0)
            },
            'needs_manual_annotation': i % 3 == 0
        }
        
        annotation_path = f"{feedback_dir}/annotations/feedback_{timestamp}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
    
    print(f"✅ Created 5 fake feedback samples in {feedback_dir}")
    return feedback_dir

def test_model_before_training():
    """Test model hiện tại"""
    print("\n🔍 Testing current model...")
    
    current_model_path = "runs/segment/success1/weights/best.pt"
    if not os.path.exists(current_model_path):
        print(f"❌ Current model not found: {current_model_path}")
        return None
    
    model = YOLO(current_model_path)
    
    # Tạo test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = model(test_image, verbose=False)
    
    print(f"📊 Current model predictions: {len(results[0].boxes) if results[0].boxes is not None else 0} detections")
    
    return {
        'model_path': current_model_path,
        'predictions': len(results[0].boxes) if results[0].boxes is not None else 0
    }

def run_transfer_learning_test():
    """Chạy transfer learning"""
    print("\n🚀 Starting Transfer Learning test...")
    
    retrainer = AutoRetrainer()
    
    # Hiển thị stats trước khi train
    stats = retrainer.get_training_stats()
    print(f"📊 Training stats: {stats}")
    
    # Chạy training với epochs thấp để test nhanh
    success = retrainer.start_yolo_training(epochs=10, model_size='n')
    
    if success:
        print("✅ Transfer Learning completed successfully!")
        return True
    else:
        print("❌ Transfer Learning failed!")
        return False

def test_model_after_training():
    """Test model sau khi training"""
    print("\n🔍 Testing new model...")
    
    new_model_path = "runs/segment/latest_retrain/best.pt"
    if not os.path.exists(new_model_path):
        print(f"❌ New model not found: {new_model_path}")
        return None
    
    model = YOLO(new_model_path)
    
    # Tạo test image giống với before
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run inference
    results = model(test_image, verbose=False)
    
    print(f"📊 New model predictions: {len(results[0].boxes) if results[0].boxes is not None else 0} detections")
    
    # Load model info
    info_path = "runs/segment/latest_retrain/model_info.json"
    model_info = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model_info = json.load(f)
    
    return {
        'model_path': new_model_path,
        'predictions': len(results[0].boxes) if results[0].boxes is not None else 0,
        'model_info': model_info
    }

def cleanup_test_data():
    """Dọn dẹp test data"""
    print("\n🧹 Cleaning up test data...")
    
    # Xóa fake feedback data
    feedback_dir = "active_learning_data"
    if os.path.exists(feedback_dir):
        # Chỉ xóa các file test
        for root, dirs, files in os.walk(feedback_dir):
            for file in files:
                if 'test_' in file:
                    os.remove(os.path.join(root, file))
        print("✅ Cleaned up test feedback data")

def main():
    """Main test function"""
    print("🎯 TRANSFER LEARNING WORKFLOW TEST")
    print("=" * 50)
    
    try:
        # 1. Tạo fake data
        create_fake_feedback_data()
        
        # 2. Test model trước training
        before_results = test_model_before_training()
        
        # 3. Chạy transfer learning
        training_success = run_transfer_learning_test()
        
        if not training_success:
            print("❌ Training failed, stopping test")
            return
        
        # 4. Test model sau training
        after_results = test_model_after_training()
        
        # 5. So sánh kết quả
        print("\n📊 COMPARISON RESULTS")
        print("=" * 30)
        
        if before_results and after_results:
            print(f"Before Training: {before_results['predictions']} predictions")
            print(f"After Training:  {after_results['predictions']} predictions")
            print(f"Model Info: {after_results.get('model_info', {})}")
            
            if 'training_type' in after_results.get('model_info', {}):
                print(f"✅ Training Type: {after_results['model_info']['training_type']}")
            
            if 'epochs' in after_results.get('model_info', {}):
                print(f"✅ Epochs: {after_results['model_info']['epochs']}")
        
        # 6. Dọn dẹp
        cleanup_choice = input("\nDọn dẹp test data? (y/n): ")
        if cleanup_choice.lower() == 'y':
            cleanup_test_data()
        
        print("\n🎉 Transfer Learning test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
