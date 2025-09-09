# -*- coding: utf-8 -*-
"""
Auto Retraining System for Diamond Detection Model
Sử dụng feedback data để train lại model theo chuẩn Roboflow
"""

import os
import json
import cv2
import numpy as np
import base64
import shutil
from datetime import datetime
import yaml

# Fix multiprocessing on Windows
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

from ultralytics import YOLO

class AutoRetrainer:
    def __init__(self, feedback_dir="active_learning_data"):
        self.feedback_dir = feedback_dir
        self.training_dir = "diamond_retrain"  # Tên folder theo chuẩn Roboflow
        self.current_model_path = "runs/segment/success1/weights/best.pt"
        
        # Tạo folder structure theo chuẩn Roboflow
        self.setup_roboflow_structure()
    
    def setup_roboflow_structure(self):
        """Tạo cấu trúc folder theo chuẩn Roboflow"""
        folders = [
            f"{self.training_dir}/train/images",
            f"{self.training_dir}/train/labels", 
            f"{self.training_dir}/valid/images",
            f"{self.training_dir}/valid/labels",
            f"{self.training_dir}/test/images",
            f"{self.training_dir}/test/labels"
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
        
        print(f"✅ Created Roboflow structure: {self.training_dir}/")
    
    def process_feedback_to_roboflow(self, task_type="detect"):
        """Xử lý feedback data thành format Roboflow chuẩn"""
        annotation_files = [f for f in os.listdir(f"{self.feedback_dir}/annotations") 
                          if f.endswith('.json')]
        
        if not annotation_files:
            print("❌ Không có feedback data để train!")
            return False
        
        print(f"📊 Processing {len(annotation_files)} feedback files to Roboflow format ({task_type})...")
        
        train_count = 0
        valid_count = 0
        test_count = 0
        
        for i, file in enumerate(annotation_files):
            try:
                # Load feedback data
                with open(f"{self.feedback_dir}/annotations/{file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Phân chia dataset theo tỉ lệ chuẩn Roboflow:
                # Train: 70%, Valid: 25%, Test: 5%
                if i % 20 == 0:  # 5% for test
                    split = "test"
                    test_count += 1
                elif i % 4 == 0:  # 25% for valid  
                    split = "valid"
                    valid_count += 1
                else:  # 70% for train
                    split = "train"
                    train_count += 1
                
                # Process image and annotations with task type
                success = self.create_roboflow_annotation(data, split, task_type)
                
                if success:
                    print(f"✅ Processed {file} → {split}")
                        
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")
                continue
        
        print(f"📈 Dataset created: {train_count} train, {valid_count} valid, {test_count} test")
        return train_count > 0
    
    def create_roboflow_annotation(self, feedback_data, split, task_type="detect"):
        """Tạo annotation Roboflow từ feedback data - support cả detect và segment"""
        try:
            # Get corrected predictions
            predictions = feedback_data.get('predictions', [])
            corrections = feedback_data.get('user_corrections', {})
            
            # Ưu tiên sử dụng true_positives nếu có
            if 'true_positives' in corrections:
                true_positives = corrections['true_positives']
            else:
                # Fallback: tính từ predictions - false_positives
                false_positives = set(corrections.get('false_positives', []))
                true_positives = [pred for i, pred in enumerate(predictions) if i not in false_positives]
            
            missed_objects = corrections.get('missed_objects', [])
            
            # Load image
            image_path = feedback_data.get('image_path')
            if not image_path or not os.path.exists(image_path):
                return False
            
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            h, w = image.shape[:2]
            
            # Create file names
            timestamp = feedback_data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            img_name = f"diamond_{timestamp}.jpg"
            label_name = f"diamond_{timestamp}.txt"
            
            # Copy image to Roboflow folder structure
            img_dst = f"{self.training_dir}/{split}/images/{img_name}"
            shutil.copy2(image_path, img_dst)
            
            # Create YOLO annotation
            annotations = []
            
            # Add TRUE POSITIVES
            for pred in true_positives:
                if task_type == "segment":
                    # Segmentation format: class x1 y1 x2 y2 x3 y3 x4 y4 (polygon)
                    x1 = pred['x'] / w
                    y1 = pred['y'] / h
                    x2 = (pred['x'] + pred['w']) / w
                    y2 = pred['y'] / h
                    x3 = (pred['x'] + pred['w']) / w
                    y3 = (pred['y'] + pred['h']) / h
                    x4 = pred['x'] / w
                    y4 = (pred['y'] + pred['h']) / h
                    
                    polygon = f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                    annotations.append(polygon)
                else:
                    # Detection format: class x_center y_center width height
                    x_center = (pred['x'] + pred['w'] / 2) / w
                    y_center = (pred['y'] + pred['h'] / 2) / h
                    width = pred['w'] / w
                    height = pred['h'] / h
                    
                    annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Add MISSED OBJECTS
            for missed in missed_objects:
                if task_type == "segment":
                    # Segmentation format
                    x1 = missed['x'] / w
                    y1 = missed['y'] / h
                    x2 = (missed['x'] + missed['w']) / w
                    y2 = missed['y'] / h
                    x3 = (missed['x'] + missed['w']) / w
                    y3 = (missed['y'] + missed['h']) / h
                    x4 = missed['x'] / w
                    y4 = (missed['y'] + missed['h']) / h
                    
                    polygon = f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
                    annotations.append(polygon)
                else:
                    # Detection format
                    x_center = (missed['x'] + missed['w'] / 2) / w
                    y_center = (missed['y'] + missed['h'] / 2) / h
                    width = missed['w'] / w
                    height = missed['h'] / h
                    
                    annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save annotation file
            label_path = f"{self.training_dir}/{split}/labels/{label_name}"
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            print(f"Created {task_type} annotation with {len(true_positives)} TP + {len(missed_objects)} missed = {len(annotations)} total")
            return True
            
        except Exception as e:
            print(f"⚠️ Error creating annotation: {e}")
            return False
    
    def create_roboflow_yaml(self):
        """Tạo data.yaml theo chuẩn Roboflow"""
        yaml_content = {
            # Đường dẫn theo chuẩn Roboflow
            'path': os.path.abspath(self.training_dir),
            'train': 'train/images',
            'val': 'valid/images', 
            'test': 'test/images',
            
            # Number of classes
            'nc': 1,
            
            # Class names
            'names': ['diamond']
        }
        
        yaml_path = f"{self.training_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"📄 Created Roboflow data.yaml: {yaml_path}")
        return yaml_path
    
    def start_yolo_training(self, epochs=100, model_size='n'):
        """Bắt đầu training theo lệnh YOLO chuẩn với Transfer Learning"""
        try:
            print("🚀 Starting YOLO training with Roboflow data...")
            
            # LUÔN SỬ DỤNG DETECTION MODEL để tránh segmentation issues
            # Detection model đơn giản hơn và ít lỗi hơn
            model_variants = {
                'n': 'yolov8n.pt',  # detection nano 
                's': 'yolov8s.pt',  # detection small
                'm': 'yolov8m.pt',  # detection medium
                'l': 'yolov8l.pt',  # detection large
                'x': 'yolov8x.pt'   # detection extra large
            }
            
            # Check for existing model first
            current_model = "runs/segment/success1/weights/best.pt"
            if os.path.exists(current_model):
                print(f"📦 Found existing model: {current_model}")
                print("� Converting to detection training for stability...")
                training_type = "transfer_learning"
            else:
                training_type = "from_pretrained"
            
            model_name = model_variants.get(model_size, 'yolov8n.pt')
            print(f"📦 Using detection model: {model_name}")
            task_type = "detect"
            
            # Process feedback data to Roboflow format with correct task type
            if not self.process_feedback_to_roboflow(task_type):
                return False
            
            # Create data.yaml
            yaml_path = self.create_roboflow_yaml()
            
            # Load và train model theo chuẩn YOLO
            model = YOLO(model_name)
            
            # Tạo timestamp cho tên training run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f'diamond_{training_type}_{timestamp}_ep{epochs}_{model_size}'
            
            # Train command với Transfer Learning parameters
            print(f"🏋️ Training: task=detect mode=train model={model_name} data={yaml_path} epochs={epochs}")
            print(f"🧠 Training type: {training_type}")
            print(f"📁 Results will be saved to: runs/detect/{run_name}")
            
            # Training parameters được tối ưu cho memory efficiency và Windows
            train_params = {
                'data': yaml_path,
                'epochs': epochs,
                'imgsz': 320,  # Giảm image size để tiết kiệm memory
                'device': 'cpu',  # Force CPU để tránh CUDA memory issues
                'project': 'runs/detect',  # Change to detect project
                'name': run_name,
                'exist_ok': True,
                'plots': False,  # Tắt plots để tiết kiệm memory
                'save': True,
                'workers': 1,  # Giảm workers để tránh multiprocessing issues
                'batch': 1,  # Batch size nhỏ để tiết kiệm memory
                'amp': False,  # Tắt automatic mixed precision
            }
            
            # Nếu là transfer learning, dùng learning rate thấp hơn
            if training_type == "transfer_learning":
                train_params.update({
                    'lr0': 0.001,      # Lower initial learning rate cho fine-tuning
                    'lrf': 0.01,       # Lower final learning rate 
                    'warmup_epochs': 3, # Fewer warmup epochs
                    'patience': 15,    # More patience for convergence
                    'close_mosaic': 10 # Close mosaic augmentation earlier
                })
                print("🔧 Using Transfer Learning parameters: lr0=0.001, lrf=0.01, patience=15")
            
            results = model.train(**train_params)
            
            # Update model nếu training thành công
            if results:
                new_model_path = f"{results.save_dir}/weights/best.pt"
                if os.path.exists(new_model_path):
                    print(f"✅ New model trained: {new_model_path}")
                    
                    # Backup old model nếu có
                    if training_type == "transfer_learning":
                        backup_dir = "runs/detect/model_backups"  # Change to detect
                        os.makedirs(backup_dir, exist_ok=True)
                        backup_path = f"{backup_dir}/backup_{timestamp}_best.pt"
                        if os.path.exists(current_model):
                            shutil.copy2(current_model, backup_path)
                            print(f"💾 Backed up old model: {backup_path}")
                    
                    # Tạo symlink hoặc copy để ap.py có thể tự động load
                    latest_model_dir = "runs/detect/latest_retrain"  # Change to detect
                    os.makedirs(latest_model_dir, exist_ok=True)
                    
                    latest_model_path = f"{latest_model_dir}/best.pt"
                    shutil.copy2(new_model_path, latest_model_path)
                    
                    # Lưu thông tin về model mới nhất
                    model_info = {
                        "timestamp": timestamp,
                        "model_path": new_model_path,
                        "model_size": model_size,
                        "epochs": epochs,
                        "training_type": training_type,
                        "base_model": model_name,
                        "training_data": f"feedback samples from {len(os.listdir(f'{self.feedback_dir}/annotations'))} files",
                        "created_at": datetime.now().isoformat()
                    }
                    
                    with open(f"{latest_model_dir}/model_info.json", 'w') as f:
                        json.dump(model_info, f, indent=2)
                    
                    print(f"🔗 Latest model symlinked to: {latest_model_path}")
                    print(f"� Model info saved to: {latest_model_dir}/model_info.json")
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def check_gpu(self):
        """Kiểm tra GPU availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_training_stats(self):
        """Lấy thống kê training data"""
        feedback_files = len([f for f in os.listdir(f"{self.feedback_dir}/annotations") 
                            if f.endswith('.json')]) if os.path.exists(f"{self.feedback_dir}/annotations") else 0
        
        train_images = len([f for f in os.listdir(f"{self.training_dir}/train/images") 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(f"{self.training_dir}/train/images") else 0
        
        valid_images = len([f for f in os.listdir(f"{self.training_dir}/valid/images") 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(f"{self.training_dir}/valid/images") else 0
        
        test_images = len([f for f in os.listdir(f"{self.training_dir}/test/images") 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(f"{self.training_dir}/test/images") else 0
        
        return {
            "feedback_files": feedback_files,
            "train_images": train_images,
            "valid_images": valid_images,
            "test_images": test_images,
            "total_samples": train_images + valid_images + test_images
        }

def main():
    """Main function để chạy retraining theo chuẩn Roboflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto Retrainer for Diamond Detection (Roboflow Style)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                       help='Model size: n(ano), s(mall), m(edium), l(arge), x(tra large)')
    parser.add_argument('--auto', action='store_true', help='Auto run without user input')
    args = parser.parse_args()
    
    retrainer = AutoRetrainer()
    
    print("🎯 DIAMOND DETECTION AUTO RETRAINER (ROBOFLOW STYLE)")
    print("=" * 60)
    
    # Get stats
    stats = retrainer.get_training_stats()
    print(f"📊 Current stats:")
    print(f"   - Feedback files: {stats['feedback_files']}")
    print(f"   - Total samples: {stats['total_samples']}")
    print(f"   - Train: {stats['train_images']}, Valid: {stats['valid_images']}, Test: {stats['test_images']}")
    
    if stats['feedback_files'] == 0:
        print("❌ No feedback data available. Please collect feedback first!")
        print("💡 Use the web interface to submit feedback on predictions")
        return
    
    # Model size info
    model_info = {
        'n': 'YOLOv8n-seg (Nano) - Fast, small model',
        's': 'YOLOv8s-seg (Small) - Balanced speed/accuracy', 
        'm': 'YOLOv8m-seg (Medium) - Better accuracy',
        'l': 'YOLOv8l-seg (Large) - High accuracy',
        'x': 'YOLOv8x-seg (X-Large) - Best accuracy'
    }
    
    print(f"🤖 Selected model: {model_info[args.model]}")
    print(f"⏱️ Training epochs: {args.epochs}")
    
    # Auto mode hoặc ask user
    if not args.auto:
        print(f"\n🤔 This will train a new model using:")
        print(f"   📁 Dataset: diamond_retrain/ (Roboflow format)")
        print(f"   🧠 Model: yolov8{args.model}-seg.pt")
        print(f"   🔄 Epochs: {args.epochs}")
        print(f"   📊 Data split: 70% train, 25% valid, 5% test")
        
        response = input(f"\nContinue with training? (y/n): ").lower()
        if response != 'y':
            print("❌ Training cancelled")
            return
    else:
        print(f"🚀 Auto mode: Starting Roboflow-style training...")
    
    # Start retraining với method mới
    success = retrainer.start_yolo_training(epochs=args.epochs, model_size=args.model)
    
    if success:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("✅ Model has been updated with user feedback")
        print("🔄 Restart the Flask application to use the new model")
        print("📁 Training results saved in runs/segment/retrain_roboflow_*/")
    else:
        print("\n❌ TRAINING FAILED!")
        print("Please check the logs for errors")
        print("💡 Make sure you have enough feedback data and GPU/CPU resources")

if __name__ == "__main__":
    main()
