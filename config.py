# Cấu hình tối ưu cho đếm kim cương 1mm
import torch

# Tự động detect device và cấu hình tương ứng
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_use_half = str(_device) == 'cuda'  # Chỉ dùng FP16 nếu có GPU
_is_cpu = str(_device) == 'cpu'

# Cấu hình tối ưu cho RTX 2060 (6GB VRAM)
if _is_cpu:
    # CPU: Giảm các tham số để tăng tốc
    YOLO_CONFIG = {
        "conf": 0.1,           # Confidence cao hơn để giảm tính toán
        "max_det": 5000,       # Giảm max detection cho CPU
        "augment": False,      # Tắt augmentation để tăng tốc
        "agnostic_nms": False, 
        "imgsz": 1024,         # Giảm kích thước ảnh cho CPU
        "half": False,         # CPU không hỗ trợ FP16
        "verbose": False
    }
    print("🖥️ Cấu hình CPU: Tối ưu tốc độ")
else:
    # GPU RTX 2060: Cấu hình cho độ chính xác tối đa
    YOLO_CONFIG = {
        "conf": 0.01,          # Confidence rất thấp để catch kim cương 1mm
        "iou": 0.3,            # IoU thấp để phân biệt kim cương gần nhau
        "max_det": 50000,      # Tối đa detection cho RTX 2060
        "augment": False,      # TẮT augmentation - YOLOv8-seg không hỗ trợ
        "agnostic_nms": False, 
        "imgsz": 1280,         # Kích thước tối ưu cho RTX 2060
        "half": True,          # FP16 cho RTX 2060
        "verbose": False,
        "retina_masks": True,  # High-quality masks
        "save": False,         # Không save để tiết kiệm memory
        "show": False
    }
    print("🚀 Cấu hình RTX 2060: Tối ưu độ chính xác (TẮT augment - không hỗ trợ)")

# Cấu hình cho morphological operations - tối ưu RTX 2060
MORPH_CONFIG = {
    "clean_kernel_size": (2, 2),    # Kernel nhỏ cho kim cương 1mm
    "erode_kernel_size": (2, 2),    # Kernel erode nhỏ
    "erode_iterations": 1,          # Ít iteration để không mất object nhỏ
    "dilate_iterations": 2,         # Ít dilation
    "distance_threshold": 0.1       # Threshold thấp hơn cho distance transform
}

# Cấu hình cho lọc object - tối ưu cho kim cương 1mm
FILTER_CONFIG = {
    "min_area_pixels": 6,              # Giảm từ 10 để catch kim cương 1mm
    "max_aspect_ratio": 2.5,           # Giảm từ 3.0 để strict hơn
    "iou_threshold": 0.25,             # Giảm từ 0.3 để phân biệt kim cương gần nhau
    "min_size_mm": 0.7,                # Giảm từ 0.8mm để catch kim cương 1mm
    "circularity_threshold": 0.3,      # Minimum circularity
    "confidence_threshold": 0.2        # Minimum confidence score
}

# Cấu hình cho image enhancement - tối ưu cho RTX 2060
ENHANCEMENT_CONFIG = {
    "clahe_clip_limit": 3.0,           # Tăng từ 2.0 cho contrast mạnh hơn
    "clahe_tile_size": (4, 4),         # Giảm từ (8,8) cho detail tốt hơn
    "gaussian_kernel": (3, 3),         
    "gaussian_sigma": 0.3,             # Giảm từ 0.5 để bảo toàn detail
    "sharpen_strength": 0.3            # Thêm sharpening
}
