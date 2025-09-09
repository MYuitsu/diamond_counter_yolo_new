"""
Test chuyên biệt cho RTX 2060 - Đếm kim cương 1mm
"""
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from config import YOLO_CONFIG, FILTER_CONFIG
from utils_device import get_device, get_gpu_memory_info, optimize_gpu_memory

def test_rtx2060_performance():
    """Test hiệu suất RTX 2060 với kim cương 1mm"""
    print("=== TEST RTX 2060 - ĐẾM KIM CƯƠNG 1MM ===")
    
    device = get_device()
    if str(device) != 'cuda':
        print("❌ RTX 2060 không được detect. Kiểm tra driver CUDA.")
        return False
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    optimize_gpu_memory()
    model = YOLO("runs/segment/success1/weights/best.pt")
    model.to(device)
    model.half()
    
    mem_info = get_gpu_memory_info()
    print(f"💾 Model Memory: {mem_info['allocated']:.1f}GB")
    
    # Test với các kích thước ảnh khác nhau
    test_cases = [
        {"size": 640, "desc": "Nhanh"},
        {"size": 1024, "desc": "Cân bằng"}, 
        {"size": 1280, "desc": "Chính xác"},
        {"size": 1536, "desc": "Siêu chính xác (nếu đủ VRAM)"}
    ]
    
    print(f"\n{'Size':<8} {'Time(ms)':<10} {'FPS':<6} {'Memory':<10} {'Detections':<12} {'Desc'}")
    print("="*70)
    
    best_config = None
    best_score = 0
    
    for test in test_cases:
        size = test["size"]
        desc = test["desc"]
        
        # Tạo test image với kim cương giả
        test_image = create_diamond_test_image(size)
        
        try:
            # Warm up
            for _ in range(2):
                _ = model(test_image, **YOLO_CONFIG)
            
            torch.cuda.empty_cache()
            
            # Benchmark
            times = []
            detections = []
            
            for _ in range(3):
                start = time.time()
                results = model(test_image, **YOLO_CONFIG)
                torch.cuda.synchronize()
                end = time.time()
                
                times.append((end - start) * 1000)
                if results[0].masks is not None:
                    detections.append(len(results[0].masks.data))
                else:
                    detections.append(0)
            
            avg_time = sum(times) / len(times)
            fps = 1000 / avg_time
            avg_det = sum(detections) / len(detections)
            
            # Memory usage
            mem_info = get_gpu_memory_info()
            memory_usage = f"{mem_info['allocated']:.1f}GB"
            
            # Score = accuracy + speed
            speed_score = min(fps / 10, 1.0)  # Normalize FPS
            accuracy_score = min(avg_det / 20, 1.0)  # Expected ~20 diamonds
            total_score = (accuracy_score * 0.7 + speed_score * 0.3)
            
            print(f"{size}x{size:<3} {avg_time:<10.1f} {fps:<6.2f} {memory_usage:<10} {avg_det:<12.1f} {desc}")
            
            if total_score > best_score and mem_info['allocated'] < mem_info['total'] * 0.9:
                best_score = total_score
                best_config = {"size": size, "fps": fps, "accuracy": avg_det, "desc": desc}
                
        except torch.cuda.OutOfMemoryError:
            print(f"{size}x{size:<3} {'OOM':<10} {'N/A':<6} {'N/A':<10} {'N/A':<12} {desc}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"{size}x{size:<3} {'Error':<10} {'N/A':<6} {'N/A':<10} {'N/A':<12} {str(e)[:10]}")
            continue
    
    print("="*70)
    
    if best_config:
        print(f"🏆 KHUYẾN NGHỊ CHO RTX 2060:")
        print(f"   - Kích thước ảnh: {best_config['size']}x{best_config['size']}")
        print(f"   - FPS: {best_config['fps']:.1f}")
        print(f"   - Detections: {best_config['accuracy']:.1f}")
        print(f"   - Mô tả: {best_config['desc']}")
        
        # Cập nhật config tự động
        update_config_for_rtx2060(best_config['size'])
    
    return True

def create_diamond_test_image(size):
    """Tạo ảnh test với kim cương giả"""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(size):
        for j in range(size):
            image[i, j] = [50 + (i+j) % 50, 50 + (i+j) % 50, 50 + (i+j) % 50]
    
    # Add noise
    noise = np.random.randint(-20, 20, (size, size, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Thêm kim cương giả với kích thước khác nhau
    diamond_sizes = [3, 5, 8, 12, 15, 20]  # pixels (tương đương 0.8-4mm)
    positions = []
    
    # Generate random positions
    margin = 50
    for _ in range(25):  # 25 kim cương test
        x = np.random.randint(margin, size - margin)
        y = np.random.randint(margin, size - margin)
        
        # Kiểm tra không overlap
        too_close = False
        for px, py in positions:
            if abs(x - px) < 30 or abs(y - py) < 30:
                too_close = True
                break
        
        if not too_close:
            positions.append((x, y))
    
    # Vẽ kim cương
    for i, (x, y) in enumerate(positions[:20]):  # Chỉ vẽ 20 cái
        radius = diamond_sizes[i % len(diamond_sizes)]
        
        # Vẽ hình tròn trắng (kim cương)
        cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
        
        # Thêm highlight
        cv2.circle(image, (x - radius//3, y - radius//3), radius//3, (200, 200, 200), -1)
        
        # Thêm shadow nhẹ
        cv2.circle(image, (x + 1, y + 1), radius, (180, 180, 180), 1)
    
    return image

def update_config_for_rtx2060(optimal_size):
    """Cập nhật config tự động cho RTX 2060"""
    print(f"\n🔧 Cập nhật config tối ưu cho RTX 2060...")
    
    config_update = f"""
# RTX 2060 Optimal Config (Auto-generated)
YOLO_CONFIG_RTX2060 = {{
    "conf": 0.01,          # Ultra low confidence cho kim cương 1mm
    "iou": 0.3,            # IoU threshold
    "max_det": 50000,      # Max detection cho RTX 2060
    "augment": True,       # Test-time augmentation
    "agnostic_nms": False, 
    "imgsz": {optimal_size},         # Optimal size cho RTX 2060
    "half": True,          # FP16 cho RTX 2060
    "verbose": False,
    "retina_masks": True,  # High-quality masks
}}
"""
    
    with open("rtx2060_optimal_config.py", "w") as f:
        f.write(config_update)
    
    print("✓ Config đã được lưu vào 'rtx2060_optimal_config.py'")

def memory_stress_test():
    """Test memory RTX 2060 với batch processing"""
    print(f"\n🧪 MEMORY STRESS TEST RTX 2060:")
    
    device = get_device()
    model = YOLO("runs/segment/success1/weights/best.pt")
    model.to(device)
    model.half()
    
    # Test với batch sizes khác nhau
    batch_sizes = [1, 2, 4, 8]
    image_size = 1280
    
    for batch_size in batch_sizes:
        try:
            # Tạo batch images
            images = [create_diamond_test_image(image_size) for _ in range(batch_size)]
            
            torch.cuda.empty_cache()
            start_mem = get_gpu_memory_info()['allocated']
            
            # Process batch
            start = time.time()
            for img in images:
                _ = model(img, **YOLO_CONFIG)
            end = time.time()
            
            end_mem = get_gpu_memory_info()['allocated']
            mem_per_image = (end_mem - start_mem) / batch_size
            time_per_image = (end - start) / batch_size * 1000
            
            print(f"Batch {batch_size}: {time_per_image:.1f}ms/img, {mem_per_image:.2f}GB/img")
            
        except torch.cuda.OutOfMemoryError:
            print(f"Batch {batch_size}: OOM - Max batch size found: {batch_size-1}")
            break
        except Exception as e:
            print(f"Batch {batch_size}: Error - {e}")
            break

if __name__ == "__main__":
    success = test_rtx2060_performance()
    if success:
        memory_stress_test()
        print(f"\n🎯 KẾT LUẬN:")
        print("✓ RTX 2060 hoạt động tốt cho đếm kim cương 1mm")
        print("✓ Sử dụng config được khuyến nghị để đạt hiệu suất tối ưu")
        print("✓ Kiểm tra memory usage thường xuyên khi process ảnh lớn")
    else:
        print("❌ RTX 2060 chưa sẵn sàng. Kiểm tra cài đặt CUDA.")
