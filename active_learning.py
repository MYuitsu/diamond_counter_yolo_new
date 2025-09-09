"""
Active Learning Tool - Thu thập dữ liệu từ predictions
Giúp cải thiện model bằng cách thêm hard examples
"""
import cv2
import numpy as np
import json
import os
from datetime import datetime
from ultralytics import YOLO
from config import YOLO_CONFIG

class ActiveLearningCollector:
    def __init__(self, model_path='runs/segment/success1/weights/best.pt'):
        self.model = YOLO(model_path)
        self.feedback_dir = "active_learning_data"
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(f"{self.feedback_dir}/images", exist_ok=True)
        os.makedirs(f"{self.feedback_dir}/annotations", exist_ok=True)
        
    def process_image_with_feedback(self, image_path, user_corrections=None):
        """
        Xử lý ảnh và thu thập feedback từ user
        
        Args:
            image_path: Đường dẫn ảnh
            user_corrections: Dict chứa corrections từ user
                {
                    'missed_objects': [(x, y, w, h), ...],  # Objects bị miss
                    'false_positives': [0, 2, 5, ...],      # Index của false positives
                    'confidence_threshold': 0.3              # Threshold tối ưu cho ảnh này
                }
        """
        # Load và predict
        image = cv2.imread(image_path)
        results = self.model(image, **YOLO_CONFIG)
        
        # Extract predictions
        predictions = []
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf, mask) in enumerate(zip(boxes, confidences, masks)):
                predictions.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'mask': mask.tolist(),
                    'is_correct': True  # Will be updated by user feedback
                })
        
        # Apply user corrections
        if user_corrections:
            # Mark false positives
            if 'false_positives' in user_corrections:
                for fp_idx in user_corrections['false_positives']:
                    if fp_idx < len(predictions):
                        predictions[fp_idx]['is_correct'] = False
            
            # Add missed objects
            if 'missed_objects' in user_corrections:
                for missed in user_corrections['missed_objects']:
                    predictions.append({
                        'box': missed,
                        'confidence': 1.0,  # Ground truth
                        'mask': None,  # Will be annotated manually
                        'is_correct': True,
                        'is_missed': True
                    })
        
        # Save feedback data
        self.save_feedback_data(image_path, image, predictions, user_corrections)
        
        return predictions
    
    def save_feedback_data(self, original_path, image, predictions, corrections):
        """Lưu dữ liệu feedback để retrain"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # Save image
        image_name = f"{base_name}_{timestamp}.jpg"
        image_path = os.path.join(self.feedback_dir, "images", image_name)
        cv2.imwrite(image_path, image)
        
        # Save annotations in YOLO format
        annotation_data = {
            'image_path': image_path,
            'original_path': original_path,
            'timestamp': timestamp,
            'predictions': predictions,
            'user_corrections': corrections,
            'needs_manual_annotation': any(p.get('is_missed', False) for p in predictions)
        }
        
        annotation_path = os.path.join(self.feedback_dir, "annotations", f"{base_name}_{timestamp}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"✅ Saved feedback data: {annotation_path}")
        
    def generate_training_suggestions(self):
        """Phân tích feedback và đề xuất training strategy"""
        annotation_files = [f for f in os.listdir(f"{self.feedback_dir}/annotations") if f.endswith('.json')]
        
        total_images = len(annotation_files)
        false_positives = 0
        missed_objects = 0
        confidence_issues = []
        
        for file in annotation_files:
            with open(f"{self.feedback_dir}/annotations/{file}", 'r') as f:
                data = json.load(f)
            
            for pred in data['predictions']:
                if not pred['is_correct']:
                    false_positives += 1
                if pred.get('is_missed', False):
                    missed_objects += 1
                if pred['confidence'] < 0.3:
                    confidence_issues.append(pred['confidence'])
        
        # Generate suggestions
        suggestions = {
            'total_feedback_images': total_images,
            'false_positive_rate': false_positives / max(1, total_images),
            'missed_detection_count': missed_objects,
            'low_confidence_predictions': len(confidence_issues),
            'recommended_actions': []
        }
        
        if false_positives > total_images * 0.1:  # >10% FP rate
            suggestions['recommended_actions'].append(
                "🔴 High false positive rate - Consider increasing confidence threshold or adding more negative examples"
            )
        
        if missed_objects > 5:
            suggestions['recommended_actions'].append(
                f"🟡 {missed_objects} missed detections - Add these as hard examples to training data"
            )
        
        if len(confidence_issues) > total_images * 0.3:
            suggestions['recommended_actions'].append(
                "🟠 Many low-confidence predictions - Model needs more training on similar examples"
            )
        
        return suggestions

def create_roboflow_format(feedback_dir):
    """Convert feedback data to Roboflow format for retraining"""
    
    output_dir = "roboflow_feedback"
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    annotation_files = [f for f in os.listdir(f"{feedback_dir}/annotations") if f.endswith('.json')]
    
    dataset_yaml = """
# Feedback Dataset for Active Learning
path: ./roboflow_feedback
train: images
val: images  # Same as train for small feedback datasets

# Classes
nc: 1  # number of classes
names: ['diamond']  # class names
"""
    
    with open(f"{output_dir}/data.yaml", 'w') as f:
        f.write(dataset_yaml)
    
    converted_count = 0
    needs_annotation = []
    
    for file in annotation_files:
        with open(f"{feedback_dir}/annotations/{file}", 'r') as f:
            data = json.load(f)
        
        if data.get('needs_manual_annotation', False):
            needs_annotation.append(data['image_path'])
            continue
        
        # Copy image
        image_name = os.path.basename(data['image_path'])
        image_dst = f"{output_dir}/images/{image_name}"
        
        import shutil
        shutil.copy2(data['image_path'], image_dst)
        
        # Create YOLO label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = f"{output_dir}/labels/{label_name}"
        
        # Get image dimensions
        image = cv2.imread(data['image_path'])
        img_h, img_w = image.shape[:2]
        
        with open(label_path, 'w') as label_file:
            for pred in data['predictions']:
                if pred['is_correct']:  # Only save correct detections
                    box = pred['box']
                    
                    # Convert to YOLO format (normalized center coordinates + width/height)
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2 / img_w
                    center_y = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Class 0 for diamond
                    label_file.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
    
    print(f"✅ Converted {converted_count} images to Roboflow format")
    print(f"📝 {len(needs_annotation)} images need manual annotation:")
    for img_path in needs_annotation:
        print(f"   - {img_path}")
    
    return output_dir

# Example usage functions
def collect_feedback_from_webapp():
    """
    Function to integrate with Flask webapp
    Collect feedback when user makes corrections
    """
    collector = ActiveLearningCollector()
    
    # Example: User uploaded image and made corrections
    user_corrections = {
        'false_positives': [1, 3],  # Detection index 1 and 3 are wrong
        'missed_objects': [
            [100, 150, 200, 250],  # x1, y1, x2, y2 of missed diamond
            [300, 400, 350, 450]   # Another missed diamond
        ],
        'confidence_threshold': 0.25  # Optimal threshold for this image
    }
    
    # Process with feedback
    predictions = collector.process_image_with_feedback(
        "path/to/user/image.jpg", 
        user_corrections
    )
    
    return predictions

if __name__ == "__main__":
    print("🎯 ACTIVE LEARNING COLLECTOR")
    print("=" * 50)
    
    # Initialize collector
    collector = ActiveLearningCollector()
    
    # Generate training suggestions from existing feedback
    suggestions = collector.generate_training_suggestions()
    
    print("📊 TRAINING SUGGESTIONS:")
    print(f"Total feedback images: {suggestions['total_feedback_images']}")
    print(f"False positive rate: {suggestions['false_positive_rate']:.2%}")
    print(f"Missed detections: {suggestions['missed_detection_count']}")
    
    print("\n🎯 RECOMMENDED ACTIONS:")
    for action in suggestions['recommended_actions']:
        print(f"  {action}")
    
    # Convert to Roboflow format if there's feedback data
    if suggestions['total_feedback_images'] > 0:
        print("\n🔄 Converting to Roboflow format...")
        output_dir = create_roboflow_format("active_learning_data")
        print(f"📁 Output: {output_dir}")
        
        print("\n📝 NEXT STEPS:")
        print("1. Upload images needing annotation to Roboflow")
        print("2. Complete manual annotations for missed objects")
        print("3. Merge with existing dataset")
        print("4. Retrain with command:")
        print(f"   yolo task=segment mode=train model=yolov8s-seg.pt data={output_dir}/data.yaml epochs=50 imgsz=1280 device=0")
