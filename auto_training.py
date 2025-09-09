"""
Auto Training Pipeline - Tự động retrain model từ feedback
"""
import os
import json
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
from datetime import datetime

class AutoTrainingPipeline:
    def __init__(self, 
                 feedback_dir="active_learning_data",
                 original_dataset_dir="diamond",  # Dataset gốc từ Roboflow
                 output_dir="enhanced_dataset"):
        
        self.feedback_dir = feedback_dir
        self.original_dataset_dir = original_dataset_dir
        self.output_dir = output_dir
        self.training_log = []
        
    def merge_datasets(self):
        """Merge feedback data với dataset gốc"""
        print("🔄 Merging feedback data with original dataset...")
        
        # Create enhanced dataset structure
        os.makedirs(f"{self.output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/val", exist_ok=True)
        
        # Copy original dataset
        if os.path.exists(self.original_dataset_dir):
            print("📁 Copying original dataset...")
            
            # Copy train images and labels
            if os.path.exists(f"{self.original_dataset_dir}/train/images"):
                for img in os.listdir(f"{self.original_dataset_dir}/train/images"):
                    shutil.copy2(
                        f"{self.original_dataset_dir}/train/images/{img}",
                        f"{self.output_dir}/images/train/{img}"
                    )
            
            if os.path.exists(f"{self.original_dataset_dir}/train/labels"):
                for lbl in os.listdir(f"{self.original_dataset_dir}/train/labels"):
                    shutil.copy2(
                        f"{self.original_dataset_dir}/train/labels/{lbl}",
                        f"{self.output_dir}/labels/train/{lbl}"
                    )
            
            # Copy validation data
            if os.path.exists(f"{self.original_dataset_dir}/valid/images"):
                for img in os.listdir(f"{self.original_dataset_dir}/valid/images"):
                    shutil.copy2(
                        f"{self.original_dataset_dir}/valid/images/{img}",
                        f"{self.output_dir}/images/val/{img}"
                    )
            
            if os.path.exists(f"{self.original_dataset_dir}/valid/labels"):
                for lbl in os.listdir(f"{self.original_dataset_dir}/valid/labels"):
                    shutil.copy2(
                        f"{self.original_dataset_dir}/valid/labels/{lbl}",
                        f"{self.output_dir}/labels/val/{lbl}"
                    )
        
        # Process feedback data
        feedback_count = self.process_feedback_data()
        
        # Create data.yaml
        self.create_data_yaml()
        
        return feedback_count
    
    def process_feedback_data(self):
        """Process và convert feedback data thành YOLO format"""
        print("🔍 Processing feedback data...")
        
        annotation_files = [f for f in os.listdir(f"{self.feedback_dir}/annotations") 
                          if f.endswith('.json')]
        
        converted_count = 0
        needs_manual = []
        
        for file in annotation_files:
            try:
                with open(f"{self.feedback_dir}/annotations/{file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if needs manual annotation
                if data.get('needs_manual_annotation', False):
                    needs_manual.append(data['image_path'])
                    continue
                
                # Process automatically correctable feedback
                if self.convert_feedback_to_yolo(data):
                    converted_count += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing {file}: {e}")
                continue
        
        print(f"✅ Converted {converted_count} feedback images")
        if needs_manual:
            print(f"📝 {len(needs_manual)} images need manual annotation:")
            for img in needs_manual[:5]:  # Show first 5
                print(f"   - {os.path.basename(img)}")
        
        return converted_count
    
    def convert_feedback_to_yolo(self, feedback_data):
        """Convert single feedback data to YOLO format"""
        try:
            image_path = feedback_data['image_path']
            if not os.path.exists(image_path):
                return False
            
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return False
                
            img_h, img_w = image.shape[:2]
            
            # Generate unique filename
            timestamp = feedback_data['timestamp']
            base_name = f"feedback_{timestamp}"
            
            # Copy image to training set
            img_dst = f"{self.output_dir}/images/train/{base_name}.jpg"
            shutil.copy2(image_path, img_dst)
            
            # Create label file
            label_path = f"{self.output_dir}/labels/train/{base_name}.txt"
            
            with open(label_path, 'w') as f:
                # Process predictions (remove false positives, keep correct ones)
                predictions = feedback_data.get('predictions', [])
                corrections = feedback_data.get('user_corrections', {})
                false_positives = corrections.get('false_positives', [])
                
                for i, pred in enumerate(predictions):
                    # Skip false positives
                    if i in false_positives:
                        continue
                    
                    # Convert box to YOLO format
                    if 'box' in pred:
                        box = pred['box']
                        x1, y1, x2, y2 = box
                        
                        # Convert to normalized center coordinates
                        center_x = (x1 + x2) / 2 / img_w
                        center_y = (y1 + y2) / 2 / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        
                        # Write YOLO label (class 0 for diamond)
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\\n")
                
                # Add missed objects if any
                missed_objects = corrections.get('missed_objects', [])
                for missed in missed_objects:
                    if 'x' in missed and 'y' in missed:
                        # Convert missed object coordinates
                        x, y, w, h = missed['x'], missed['y'], missed['w'], missed['h']
                        center_x = (x + w/2) / img_w
                        center_y = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\\n")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error converting feedback: {e}")
            return False
    
    def create_data_yaml(self):
        """Create data.yaml for enhanced dataset"""
        yaml_content = f"""
# Enhanced Dataset with Active Learning Feedback
path: ./{self.output_dir}
train: images/train
val: images/val

# Classes
nc: 1
names: ['diamond']

# Training info
original_dataset: {self.original_dataset_dir}
feedback_integrated: true
creation_date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(f"{self.output_dir}/data.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Created data.yaml for enhanced dataset")
    
    def auto_train(self, 
                   model_size='s',  # n, s, m, l, x
                   epochs=50,
                   imgsz=1280,
                   device='0'):
        """Automatically start training with enhanced dataset"""
        
        print(f"🚀 Starting automatic training...")
        print(f"Model: YOLOv8{model_size}-seg")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Device: {device}")
        
        # Training command
        cmd = [
            'yolo',
            'task=segment',
            'mode=train',
            f'model=yolov8{model_size}-seg.pt',
            f'data={self.output_dir}/data.yaml',
            f'epochs={epochs}',
            f'imgsz={imgsz}',
            f'device={device}',
            'save_period=10',  # Save every 10 epochs
            'patience=20',     # Early stopping
            'optimizer=AdamW',
            'lr0=0.001',      # Lower learning rate for fine-tuning
            'cos_lr=True'     # Cosine learning rate scheduler
        ]
        
        try:
            # Log training start
            self.training_log.append({
                'timestamp': datetime.now().isoformat(),
                'command': ' '.join(cmd),
                'status': 'started'
            })
            
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                self.training_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'output': result.stdout[-500:]  # Last 500 chars
                })
                
                # Find the new model
                runs_dir = "runs/segment"
                if os.path.exists(runs_dir):
                    train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
                    if train_dirs:
                        latest_train = max(train_dirs)
                        new_model_path = f"{runs_dir}/{latest_train}/weights/best.pt"
                        print(f"🎯 New model saved: {new_model_path}")
                        return new_model_path
                
            else:
                print("❌ Training failed!")
                print(result.stderr)
                self.training_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed',
                    'error': result.stderr
                })
                return None
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None
    
    def evaluate_new_model(self, model_path):
        """Evaluate new model performance"""
        print(f"📊 Evaluating new model: {model_path}")
        
        try:
            # Load new model
            model = YOLO(model_path)
            
            # Validate on validation set
            results = model.val(data=f"{self.output_dir}/data.yaml")
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50 if hasattr(results, 'box') else 'N/A',
                'mAP50-95': results.box.map if hasattr(results, 'box') else 'N/A',
                'precision': results.box.mp if hasattr(results, 'box') else 'N/A',
                'recall': results.box.mr if hasattr(results, 'box') else 'N/A'
            }
            
            print("📈 New Model Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value}")
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            return None
    
    def run_full_pipeline(self, auto_train=True):
        """Run complete active learning pipeline"""
        print("🎯 ACTIVE LEARNING PIPELINE")
        print("=" * 50)
        
        # Step 1: Merge datasets
        feedback_count = self.merge_datasets()
        
        if feedback_count == 0:
            print("⚠️ No feedback data to process")
            return None
        
        print(f"✅ Integrated {feedback_count} feedback examples")
        
        # Step 2: Auto train if requested
        if auto_train:
            new_model_path = self.auto_train()
            
            if new_model_path:
                # Step 3: Evaluate
                metrics = self.evaluate_new_model(new_model_path)
                
                # Step 4: Generate report
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'feedback_integrated': feedback_count,
                    'new_model_path': new_model_path,
                    'metrics': metrics,
                    'training_log': self.training_log
                }
                
                # Save report
                with open(f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                    json.dump(report, f, indent=2)
                
                print("🎉 Active learning pipeline completed!")
                print(f"📊 Report saved: training_report_*.json")
                print(f"🎯 New model: {new_model_path}")
                
                return new_model_path
        
        return None

if __name__ == "__main__":
    print("🚀 ACTIVE LEARNING TRAINING PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AutoTrainingPipeline()
    
    # Check if feedback data exists
    feedback_dir = "active_learning_data/annotations"
    if not os.path.exists(feedback_dir) or not os.listdir(feedback_dir):
        print("⚠️ No feedback data found")
        print("📝 To collect feedback data:")
        print("   1. Run webapp with feedback collection")
        print("   2. Users make corrections on predictions")
        print("   3. System automatically saves feedback")
        print("   4. Run this pipeline to retrain")
    else:
        # Run pipeline
        new_model = pipeline.run_full_pipeline(auto_train=True)
        
        if new_model:
            print(f"\\n🎯 NEXT STEPS:")
            print(f"1. Test new model: python ap.py")
            print(f"2. Compare with old model performance")
            print(f"3. Deploy if metrics improved")
            print(f"4. Continue collecting feedback")
        else:
            print("\\n⚠️ Pipeline completed but no new model created")
    
    print(f"\\n💡 AUTOMATION TIP:")
    print(f"   Schedule this script to run daily/weekly:")
    print(f"   python auto_training.py")
