from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import math
import json
import os
from datetime import datetime
import threading
import subprocess

app = Flask(__name__, static_folder='static')

CORS(app, resources={r"/*": {"origins": "*"}})

def get_latest_model_path():
    """Tự động tìm và load model mới nhất từ training"""
    # Thứ tự ưu tiên:
    # 1. Model retrain mới nhất từ latest_retrain
    # 2. Model gốc từ success1
    
    latest_retrain_path = "runs/segment/latest_retrain/best.pt"
    original_model_path = "runs/segment/success1/weights/best.pt"
    
    if os.path.exists(latest_retrain_path):
        # Kiểm tra thông tin model
        info_path = "runs/segment/latest_retrain/model_info.json"
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                print(f"📦 Loading RETRAINED model: {model_info['timestamp']} (Epochs: {model_info['epochs']}, Size: {model_info['model_size']})")
                return latest_retrain_path
            except:
                pass
        else:
            print(f"📦 Loading retrained model: {latest_retrain_path}")
            return latest_retrain_path
    
    if os.path.exists(original_model_path):
        print(f"📦 Loading ORIGINAL model: {original_model_path}")
        return original_model_path
    
    # Fallback tìm model mới nhất trong runs/segment
    segment_dir = "runs/segment"
    if os.path.exists(segment_dir):
        retrain_dirs = [d for d in os.listdir(segment_dir) if d.startswith('diamond_retrain_')]
        if retrain_dirs:
            # Sắp xếp theo timestamp
            retrain_dirs.sort(reverse=True)
            latest_dir = retrain_dirs[0]
            fallback_path = f"{segment_dir}/{latest_dir}/weights/best.pt"
            if os.path.exists(fallback_path):
                print(f"📦 Loading FALLBACK retrained model: {fallback_path}")
                return fallback_path
    
    # Cuối cùng fallback về yolo default
    print(f"⚠️ No custom model found, using default YOLOv8")
    return "yolov8n-seg.pt"

# Load model with auto-detection of latest retrained version
model_path = get_latest_model_path()
model = YOLO(model_path)

all_detections = []

TARGET_FINAL_SIZE = 1280 # Kích thước mục tiêu cho ảnh cuối cùng

# Initialize active learning collector
class ActiveLearningCollector:
    def __init__(self):
        self.feedback_dir = "active_learning_data"
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(f"{self.feedback_dir}/images", exist_ok=True)
        os.makedirs(f"{self.feedback_dir}/annotations", exist_ok=True)
    
    def save_user_feedback(self, image_data, predictions, user_corrections):
        """Save user feedback for retraining - bao gồm true positives, false positives và missed objects"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_name = f"feedback_{timestamp}.jpg"
        image_path = os.path.join(self.feedback_dir, "images", image_name)
        
        # Decode base64 image
        if isinstance(image_data, str):
            img_data = base64.b64decode(image_data)
            with open(image_path, 'wb') as f:
                f.write(img_data)
        
        # Tính toán true positives nếu chưa có
        if 'true_positives' not in user_corrections:
            false_positive_indices = set(user_corrections.get('false_positives', []))
            true_positives = [pred for i, pred in enumerate(predictions) 
                            if i not in false_positive_indices]
            user_corrections['true_positives'] = true_positives
        
        # Đếm các loại annotations
        tp_count = len(user_corrections.get('true_positives', []))
        fp_count = len(user_corrections.get('false_positives', []))
        missed_count = len(user_corrections.get('missed_objects', []))
        total_ground_truth = tp_count + missed_count
        
        # Save annotations với thông tin đầy đủ
        annotation_data = {
            'image_path': image_path,
            'timestamp': timestamp,
            'predictions': predictions,
            'user_corrections': user_corrections,
            'annotation_stats': {
                'true_positives': tp_count,
                'false_positives': fp_count, 
                'missed_objects': missed_count,
                'total_predictions': len(predictions),
                'total_ground_truth': total_ground_truth
            },
            'needs_manual_annotation': missed_count > 0
        }
        
        annotation_path = os.path.join(self.feedback_dir, "annotations", f"feedback_{timestamp}.json")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2)
        
        return annotation_path

# Initialize collector
feedback_collector = ActiveLearningCollector()

def resize_and_pad_image(image, target_size):
    h, w = image.shape[:2]
    ratio = min(target_size / w, target_size / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    padded_image = np.full((target_size, target_size, 3), 0, dtype=np.uint8) # Ảnh đen

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    return padded_image, x_offset, y_offset, new_w, new_h

def draw_boxes_with_numbers(image, results, filtered_indexes, font_size):
    for i, box in enumerate(results[0].boxes):
        if i not in filtered_indexes:
            continue    
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{i + 1}"
        font_scale = font_size / 20.0
        font_thickness = max(2, int(font_scale * 2))

        # Tọa độ chữ và tính kích thước
        org = (x1 + 5, y1 + 20)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Vẽ background đen cho text để tăng độ tương phản
        bg_x1, bg_y1 = org[0] - 3, org[1] - text_size[1] - 3
        bg_x2, bg_y2 = org[0] + text_size[0] + 3, org[1] + 5
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Vẽ viền đen dày hơn
        cv2.putText(image, label, org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 4)

        # Vẽ chữ màu vàng sáng đè lên để dễ nhìn
        cv2.putText(image, label, org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)
    return image

def detect_shape(contour, circle_threshold=0.75):
    if len(contour) < 5:
        return "Không rõ"
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Không rõ"
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Nếu contour đủ điểm thì mới fit ellipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (x, y), (MA, ma) = ellipse[:2]
        ratio = min(MA, ma) / max(MA, ma)

        if ratio > circle_threshold and circularity > 0.4:
            return "Tròn"
        elif 0.2 < ratio:
            return "Tam giác"  
    return "Không rõ"



def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def remove_duplicate_boxes(details, iou_threshold=0.5):
    unique = []
    added_boxes = []
    for d in details:
        box = [d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]]
        is_duplicate = False
        for ub in added_boxes:
            if iou(box, ub) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(d)
            added_boxes.append(box)
    return unique

def analyze_diamond_sizes(results, tray_width, tray_length, region_w, region_h, min_size_mm, circle_threshold=0.75):
    details = []
    indexes = []
    masks = results[0].masks

    if masks is None:
        return details, indexes

    mm_per_pixel_x = tray_width / region_w
    mm_per_pixel_y = tray_length / region_h

    for i, mask in enumerate(masks.data):
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        else:
            mask_np = mask.astype(np.uint8) * 255


        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        width_mm = w * mm_per_pixel_x
        height_mm = h * mm_per_pixel_y
        area_mm2 = area * mm_per_pixel_x * mm_per_pixel_y
        diameter_mm = (width_mm + height_mm) / 2

        if diameter_mm >= min_size_mm:
            shape = detect_shape(contour, circle_threshold)
            details.append({
                "index": i + 1,
                "width": width_mm,
                "height": height_mm,
                "area": area_mm2,
                "diameter": diameter_mm,
                "shape": shape,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })
            indexes.append(i)

    # Loại trùng lặp
    filtered_details = remove_duplicate_boxes(details)
    filtered_indexes = [d["index"] - 1 for d in filtered_details]
    return filtered_details, filtered_indexes

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"})
    
    mode = request.args.get("mode", "box")
    tray_length = float(request.form.get("tray_length", 150))
    tray_width = float(request.form.get("tray_width", 80))
    min_size_mm = float(request.form.get("min_size", 1.0))
    region_x = int(request.form.get("region_x", 0))
    region_y = int(request.form.get("region_y", 0))
    region_w = int(request.form.get("region_w", 640))
    region_h = int(request.form.get("region_h", 640))
    font_size = int(request.form.get("font_size", 16))
    circle_threshold = float(request.form.get("circle_threshold", 0.75))

    img_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    # Lưu kích thước ảnh gốc
    original_img_h, original_img_w = image.shape[:2]

    crop = image[region_y:region_y+region_h, region_x:region_x+region_w]
    _, buffer_original_crop = cv2.imencode('.jpg', crop)
    encoded_original_crop = base64.b64encode(buffer_original_crop).decode('utf-8')
    resized_crop = cv2.resize(crop, (1280, 1280))


    results = model(resized_crop, conf=0.1, max_det=10000)


    details, filtered_indexes = analyze_diamond_sizes(
        results, tray_width, tray_length, region_w, region_h, min_size_mm, circle_threshold
    )
    count = len(filtered_indexes)

    
    annotated = draw_boxes_with_numbers(resized_crop.copy(), results, filtered_indexes, font_size)

    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    global all_detections
    masks = results[0].masks
    if masks is not None:
        for i, mask in enumerate(masks.data):
            if hasattr(mask, 'cpu'):
                mask_np = mask.cpu().numpy().astype(np.uint8)  # 0/1
            else:
                mask_np = mask.astype(np.uint8) * 255
            all_detections.append({
                "mask": mask_np,
                "region_x": region_x,
                "region_y": region_y,
                "region_w": region_w,
                "region_h": region_h,
                "original_img_w": original_img_w,  # Thêm kích thước ảnh gốc
                "original_img_h": original_img_h   # Thêm kích thước ảnh gốc
                #"original_crop": encoded_original_crop # Store original cropped image
            })


    return jsonify({
        "diamond_count": count,
        "details": details,
        "annotated_image": encoded_img,

    })

@app.route("/reset", methods=["POST"])
def reset():
    global all_detections
    all_detections = []
    return jsonify({"status": "reset"})

@app.route("/final_result", methods=["GET", "POST"])
def final_result():
    global all_detections
    valid_detections = [d for d in all_detections if 'mask' in d]
    if not valid_detections:
        blank = np.zeros((1280, 1280, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"total_diamonds": 0, "details": [], "annotated_image": encoded_img})

    # Lấy thông tin khay từ form (do frontend truyền lên)
    tray_width = float(request.form.get("tray_width", 80))
    tray_length = float(request.form.get("tray_length", 150))
    region_x_khay = int(request.form.get("region_x", 0))
    region_y_khay = int(request.form.get("region_y", 0))
    region_w_khay = int(request.form.get("region_w", 640))
    region_h_khay = int(request.form.get("region_h", 640))
    min_size_mm = float(request.form.get("min_size", 1.0))
    font_size = int(request.form.get("font_size", 16))
    circle_threshold = float(request.form.get("circle_threshold", 0.75))


    # Lấy ảnh gốc từ request.files
    original_file = request.files.get("original_image")
    original_image_decoded = None
    original_image_padded = None
    pad_x_offset_final = 0
    pad_y_offset_final = 0
    content_w_final = 0
    content_h_final = 0

    if original_file:
        img_np_original = np.frombuffer(original_file.read(), np.uint8)
        original_image_raw = cv2.imdecode(img_np_original, cv2.IMREAD_COLOR)
        
        # Resize và pad ảnh gốc về kích thước TARGET_FINAL_SIZE
        original_image_padded, pad_x_offset_final, pad_y_offset_final, content_w_final, content_h_final = \
            resize_and_pad_image(original_image_raw, TARGET_FINAL_SIZE)
        original_image_decoded = original_image_padded # Sử dụng ảnh đã được pad

    # Lấy kích thước ảnh gốc BAN ĐẦU (trước khi crop và resize ở frontend, nhưng là kích thước ảnh tải lên ban đầu)
    # Ta dùng original_image_raw.shape từ ảnh vừa nhận ở backend
    original_img_w_at_backend_entry = original_image_raw.shape[1] if original_image_raw is not None else 0
    original_img_h_at_backend_entry = original_image_raw.shape[0] if original_image_raw is not None else 0

    # Tính toán padding và kích thước nội dung ở frontend (trên canvas 640x640)
    # Dựa trên kích thước ảnh gốc ban đầu (original_img_w_at_backend_entry, original_img_h_at_backend_entry)
    ratio_frontend_content_scale = min(640 / original_img_w_at_backend_entry, 640 / original_img_h_at_backend_entry) if original_img_w_at_backend_entry and original_img_h_at_backend_entry else 1
    frontend_actual_content_w = int(original_img_w_at_backend_entry * ratio_frontend_content_scale)
    frontend_actual_content_h = int(original_img_h_at_backend_entry * ratio_frontend_content_scale)
    frontend_pad_x_offset = (640 - frontend_actual_content_w) // 2
    frontend_pad_y_offset = (640 - frontend_actual_content_h) // 2

    # Tính toán tỉ lệ scale từ kích thước nội dung thực tế của frontend (ảnh gốc không đệm trên canvas 640x640)
    # sang kích thước ảnh gốc đã được resize và pad về TARGET_FINAL_SIZE (1280x1280) ở backend
    scale_frontend_content_to_final_x = content_w_final / frontend_actual_content_w if frontend_actual_content_w else 1
    scale_frontend_content_to_final_y = content_h_final / frontend_actual_content_h if frontend_actual_content_h else 1

    # Tổng hợp mask trên toàn ảnh lớn
    big_h = max([d["region_y"] + d["region_h"] for d in valid_detections])
    big_w = max([d["region_x"] + d["region_w"] for d in valid_detections])
    accumulation_mask = np.zeros((big_h, big_w), dtype=np.uint8)

    for det in valid_detections:
        mask = det["mask"]
        region_x = det["region_x"]
        region_y = det["region_y"]
        region_w = det["region_w"]
        region_h = det["region_h"]
        mask_resized = cv2.resize(mask, (region_w, region_h), interpolation=cv2.INTER_NEAREST)
        accumulation_mask[region_y:region_y+region_h, region_x:region_x+region_w] += (mask_resized > 0).astype(np.uint8)

    # Chỉ lấy vùng khay (theo khung hồng) để phân tích
    mask_khay = accumulation_mask[region_y_khay:region_y_khay+region_h_khay, region_x_khay:region_x_khay+region_w_khay]
    mask_khay = (mask_khay > 0).astype(np.uint8) * 255

    # Erode mask để tách vùng dính nhẹ
    kernel = np.ones((3, 3), np.uint8)
    mask_khay_eroded = cv2.erode(mask_khay, kernel, iterations=1)

    # Distance transform trên mask đã erode
    dist = cv2.distanceTransform(mask_khay_eroded, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0) # Changed threshold to 0.2
    sure_fg = np.uint8(sure_fg)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate((mask_khay > 0).astype(np.uint8), kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    mask_khay_color = cv2.cvtColor(mask_khay, cv2.COLOR_GRAY2BGR)
    


  
    markers = cv2.watershed(mask_khay_color, markers)


    # Tách từng mask viên kim cương trong vùng khay
    masks_list = []
    for marker_val in np.unique(markers):
        if marker_val <= 1:
            continue
        mask_single = (markers == marker_val).astype(np.uint8)
        masks_list.append(mask_single)

    # Tạo object giả YOLO results
    class DummyMask:
        def __init__(self, data):
            self.data = data

    class DummyResult:
        def __init__(self, masks):
            self.masks = masks

    dummy_masks = DummyMask(data=[np.array(m) for m in masks_list])
    dummy_result = DummyResult(masks=dummy_masks)

    # Gọi lại analyze_diamond_sizes với đúng tỉ lệ mm/pixel của khay
    details, filtered_indexes = analyze_diamond_sizes(
        [dummy_result], tray_width, tray_length, region_w_khay, region_h_khay, min_size_mm, circle_threshold
    )

    # Ảnh để vẽ (ảnh gốc đã resize KHÔNG ĐỆM)
    annotated_image_to_return = original_image_padded.copy() if original_image_padded is not None else np.zeros((TARGET_FINAL_SIZE, TARGET_FINAL_SIZE, 3), dtype=np.uint8)

    # Vẽ lên ảnh gốc đã resize và pad (TARGET_FINAL_SIZE x TARGET_FINAL_SIZE)
    for i, d in enumerate(details):
        # d["x"], d["y"], d["w"], d["h"] là tọa độ trên mask_khay (vùng đã crop và xử lý)
        # mask_khay được lấy từ accumulation_mask, mà accumulation_mask được tạo trên hệ tọa độ 640x640 đã đệm

        # Bước 1: Chuyển tọa độ từ mask_khay (trên vùng 640x640 đã đệm) về hệ tọa độ của ảnh gốc 640x640 (nội dung thực, không đệm)
        x_on_frontend_content_640 = d["x"] + region_x_khay - frontend_pad_x_offset
        y_on_frontend_content_640 = d["y"] + region_y_khay - frontend_pad_y_offset
        w_on_frontend_content_640 = d["w"]
        h_on_frontend_content_640 = d["h"]
        
        # Bước 2: Scale từ tọa độ nội dung thực tế của frontend (640x640) 
        # lên tọa độ trên ảnh gốc đã được resize và pad về TARGET_FINAL_SIZE (1280x1280) ở backend
        # Sau đó bù thêm offset của padding ở backend
        final_x = int(x_on_frontend_content_640 * scale_frontend_content_to_final_x) + pad_x_offset_final
        final_y = int(y_on_frontend_content_640 * scale_frontend_content_to_final_y) + pad_y_offset_final
        final_w = int(w_on_frontend_content_640 * scale_frontend_content_to_final_x)
        final_h = int(h_on_frontend_content_640 * scale_frontend_content_to_final_y)

        cv2.rectangle(annotated_image_to_return, (final_x, final_y), (final_x + final_w, final_y + final_h), (0, 255, 0), 2)
        label = f"{i + 1}"
        font_scale = font_size / 30.0
        font_thickness = max(2, int(font_scale * 3))
        org = (final_x + 5, final_y + 20)

        # Tính kích thước text để vẽ background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Vẽ background đen cho text để tăng độ tương phản
        bg_x1, bg_y1 = org[0] - 3, org[1] - text_size[1] - 3
        bg_x2, bg_y2 = org[0] + text_size[0] + 3, org[1] + 5
        cv2.rectangle(annotated_image_to_return, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        # Viền đen dày hơn
        cv2.putText(annotated_image_to_return, label, org,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 4)

        # Chữ màu vàng sáng để dễ nhìn
        cv2.putText(annotated_image_to_return, label, org,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)

                    

    cv2.imwrite("accumulation_mask_debug.png", accumulation_mask)
    cv2.imwrite("watershed_labels.png", ((markers > 1).astype(np.uint8) * 255))
    
    # Ghép ảnh gốc và ảnh đã annotated (ảnh đã được đánh dấu trực tiếp lên ảnh gốc)
    # Không cần ghép ảnh nữa vì đã vẽ trực tiếp lên ảnh gốc
    combined_image = annotated_image_to_return

    _, buffer = cv2.imencode('.jpg', combined_image)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "total_diamonds": len(details),
        "details": details,
        "annotated_image": encoded_img,
    })

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    """
    Endpoint để nhận feedback từ user về predictions
    """
    try:
        data = request.json
        image_data = data.get('image_data')
        predictions = data.get('predictions', [])
        corrections = data.get('corrections', {})
        
        # Save feedback
        annotation_path = feedback_collector.save_user_feedback(
            image_data, predictions, corrections
        )
        
        return jsonify({
            "status": "success",
            "message": "Feedback saved for retraining",
            "annotation_path": annotation_path
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error: {str(e)}"
        }), 500

@app.route("/get_training_suggestions", methods=["GET"])
def get_training_suggestions():
    """Get suggestions for retraining based on feedback"""
    try:
        annotation_files = [f for f in os.listdir(f"{feedback_collector.feedback_dir}/annotations") 
                          if f.endswith('.json')]
        
        if not annotation_files:
            return jsonify({"message": "No feedback data available"})
        
        total_feedback = len(annotation_files)
        false_positives = 0
        missed_objects = 0
        
        for file in annotation_files[-10:]:  # Last 10 feedback
            try:
                with open(f"{feedback_collector.feedback_dir}/annotations/{file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                corrections = data.get('user_corrections', {})
                false_positives += len(corrections.get('false_positives', []))
                missed_objects += len(corrections.get('missed_objects', []))
                        
            except:
                continue
        
        suggestions = []
        
        if false_positives > total_feedback * 0.15:
            suggestions.append("⚠️ High false positive rate - increase confidence threshold")
        
        if missed_objects > 5:
            suggestions.append(f"🎯 {missed_objects} missed detections - add to training data")
        
        return jsonify({
            "total_feedback": total_feedback,
            "false_positives": false_positives,
            "missed_objects": missed_objects,
            "suggestions": suggestions,
            "retrain_command": "yolo task=segment mode=train model=yolov8s-seg.pt data=roboflow_feedback/data.yaml epochs=50"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/start_retraining", methods=["POST"])
def start_retraining():
    """Bắt đầu quá trình retraining model với feedback data"""
    try:
        # Kiểm tra có feedback data không
        annotation_files = [f for f in os.listdir(f"{feedback_collector.feedback_dir}/annotations") 
                          if f.endswith('.json')]
        
        if not annotation_files:
            return jsonify({
                "status": "error",
                "message": "Không có feedback data để train! Vui lòng thu thập feedback trước."
            })
        
        # Get training parameters
        data = request.json or {}
        epochs = data.get('epochs', 100)  # Tăng default epochs theo chuẩn
        model_size = data.get('model_size', 'n')  # n, s, m, l, x
        
        # Start retraining in background thread
        def run_retraining():
            try:
                # Sử dụng memory-efficient training script
                result = subprocess.run([
                    'python', 'memory_efficient_train.py', 
                    '--epochs', str(epochs),
                    '--model', str(model_size),
                    '--auto'
                ], 
                capture_output=True, 
                text=True, 
                encoding='utf-8',
                errors='replace',
                cwd=os.getcwd())
                
                print(f"Memory-efficient training output: {result.stdout}")
                if result.stderr:
                    print(f"Training warnings: {result.stderr}")
                if result.returncode == 0:
                    print("Training completed successfully!")
                else:
                    print(f"Training failed with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Retraining thread error: {e}")
        
        # Start background thread
        training_thread = threading.Thread(target=run_retraining)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "success",
            "message": f"🚀 Bắt đầu training theo chuẩn Roboflow với {len(annotation_files)} feedback samples",
            "feedback_count": len(annotation_files),
            "epochs": epochs,
            "model_size": f"yolov8{model_size}-seg.pt",
            "dataset_format": "Roboflow (70% train, 25% valid, 5% test)",
            "note": "Training đang chạy trong background. Kiểm tra terminal để xem tiến trình."
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi bắt đầu training: {str(e)}"
        })

@app.route("/reload_model", methods=["POST"])
def reload_model():
    """Reload model để sử dụng version mới nhất sau khi training xong"""
    try:
        global model
        
        # Tìm và load model mới nhất
        new_model_path = get_latest_model_path()
        
        # Backup thông tin model cũ
        old_model_info = {
            'path': getattr(model, 'ckpt_path', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Load model mới
        model = YOLO(new_model_path)
        
        # Lấy thông tin model mới
        model_info = {}
        info_path = "runs/segment/latest_retrain/model_info.json"
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
            except:
                pass
        
        print(f"🔄 Model reloaded: {new_model_path}")
        
        return jsonify({
            "status": "success", 
            "message": "Model đã được reload thành công!",
            "model_path": new_model_path,
            "model_info": model_info,
            "old_model": old_model_info['path'],
            "reload_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi khi reload model: {str(e)}"
        })

@app.route("/training_status", methods=["GET"])
def training_status():
    """Kiểm tra trạng thái training"""
    try:
        # Kiểm tra các file training gần đây
        runs_dir = "runs/segment"
        status_info = {
            "current_model": model_path,
            "training_runs": [],
            "latest_retrain": None
        }
        
        if os.path.exists(runs_dir):
            subdirs = [d for d in os.listdir(runs_dir) if d.startswith('diamond_retrain_')]
            if subdirs:
                # Sắp xếp theo timestamp
                subdirs.sort(reverse=True)
                
                for subdir in subdirs[:5]:  # Top 5 latest
                    weights_path = os.path.join(runs_dir, subdir, "weights", "best.pt")
                    if os.path.exists(weights_path):
                        # Parse thông tin từ tên folder
                        parts = subdir.split('_')
                        if len(parts) >= 4:
                            timestamp = f"{parts[2]}_{parts[3]}"
                            epochs = parts[4].replace('ep', '') if len(parts) > 4 else 'unknown'
                            model_size = parts[5] if len(parts) > 5 else 'unknown'
                            
                            status_info["training_runs"].append({
                                "name": subdir,
                                "timestamp": timestamp,
                                "epochs": epochs,
                                "model_size": model_size,
                                "path": weights_path,
                                "is_current": weights_path == model_path
                            })
                
                if subdirs:
                    latest_run = subdirs[0]
                    latest_weights = os.path.join(runs_dir, latest_run, "weights", "best.pt")
                    
                    if os.path.exists(latest_weights):
                        status_info["latest_retrain"] = latest_run
                        status_info["status"] = "completed"
                        status_info["message"] = f"✅ Latest training: {latest_run}"
                    else:
                        status_info["status"] = "running"
                        status_info["message"] = f"🏋️ Training in progress: {latest_run}"
        
        # Kiểm tra model info
        info_path = "runs/segment/latest_retrain/model_info.json"
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                status_info["current_model_info"] = json.load(f)
        
        if not status_info["training_runs"]:
            status_info["status"] = "idle"
            status_info["message"] = "Chưa có training nào được thực hiện"
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Lỗi kiểm tra trạng thái: {str(e)}"
        })

@app.route("/model_info", methods=["GET"])
def model_info():
    """Lấy thông tin model hiện tại"""
    try:
        info = {
            "current_model_path": model_path,
            "model_type": "retrained" if "retrain" in model_path else "original"
        }
        
        # Thông tin chi tiết nếu có
        info_path = "runs/segment/latest_retrain/model_info.json"
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info["retrain_info"] = json.load(f)
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("🚀 DIAMOND COUNTER WITH ACTIVE LEARNING")
    print("📊 Feedback collector ready")
    print(f"🤖 Current model: {model_path}")
    
    # Hiển thị thông tin model retrain nếu có
    info_path = "runs/segment/latest_retrain/model_info.json"
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"✨ Using RETRAINED model:")
            print(f"   📅 Trained: {model_info['timestamp']}")
            print(f"   🧠 Model size: {model_info['model_size']}")
            print(f"   🔄 Epochs: {model_info['epochs']}")
            print(f"   📊 Data: {model_info['training_data']}")
        except:
            print("✨ Using retrained model (info file corrupted)")
    else:
        print("📦 Using original model")
    
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)



