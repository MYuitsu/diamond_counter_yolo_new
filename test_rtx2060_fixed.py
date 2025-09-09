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
    print("🚀 RTX 2060 PERFORMANCE TEST")
    print("=" * 50)
    
    # Load model
    model = YOLO('runs/segment/success1/weights/best.pt')
    device = get_device()
    model.to(device)
    
    print(f"📱 Device: {device}")
    print(f"📊 PyTorch CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        optimize_gpu_memory()
    
    # Test configurations
    test_configs = [
        {"size": 640, "desc": "Fast"},
        {"size": 896, "desc": "Balanced"}, 
        {"size": 1280, "desc": "High Quality"},
        {"size": 1536, "desc": "Ultra (risky)"}
    ]
    
    print(f"\n📊 PERFORMANCE BENCHMARK")
    print("-" * 70)
    print(f"{'Size':<8} {'Time (ms)':<10} {'FPS':<6} {'Memory':<10} {'Detections':<12} {'Mode'}")
    print("-" * 70)
    
    best_config = None
    best_score = 0
    
    for config in test_configs:
        size = config["size"]
        desc = config["desc"]
        
        # Tạo test image với kim cương giả
        test_image = create_diamond_test_image(size)
        
        try:
            # Warm up
            for _ in range(2):
                _ = model(test_image, **YOLO_CONFIG)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Benchmark
            times = []
            detections = []
            
            for _ in range(3):
                start = time.time()
                results = model(test_image, **YOLO_CONFIG)
                if torch.cuda.is_available():
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
            if torch.cuda.is_available():
                mem_info = get_gpu_memory_info()
                memory_usage = f"{mem_info['allocated']:.1f}GB"
            else:
                memory_usage = "CPU"
            
            # Score = accuracy + speed
            speed_score = min(fps / 10, 1.0)  # Normalize FPS
            accuracy_score = min(avg_det / 20, 1.0)  # Expected ~20 diamonds
            total_score = (accuracy_score * 0.7 + speed_score * 0.3)
            
            print(f"{size}x{size:<3} {avg_time:<10.1f} {fps:<6.2f} {memory_usage:<10} {avg_det:<12.1f} {desc}")
            
            gpu_safe = True
            if torch.cuda.is_available():
                gpu_safe = mem_info['allocated'] < mem_info['total'] * 0.9
                
            if total_score > best_score and gpu_safe:
                best_score = total_score
                best_config = {"size": size, "fps": fps, "accuracy": avg_det, "desc": desc}
                
        except torch.cuda.OutOfMemoryError:
            print(f"{size}x{size:<3} {'OOM':<10} {'N/A':<6} {'N/A':<10} {'N/A':<12} {desc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"{size}x{size:<3} {'Error':<10} {'N/A':<6} {'N/A':<10} {'N/A':<12} {str(e)[:10]}")
    
    # Test memory stress
    print(f"\n🧠 MEMORY STRESS TEST")
    test_memory_limits()
    
    # Recommend best config
    if best_config:
        print(f"\n🎯 OPTIMAL CONFIGURATION")
        print(f"Size: {best_config['size']}x{best_config['size']}")
        print(f"Mode: {best_config['desc']}")
        print(f"FPS: {best_config['fps']:.2f}")
        print(f"Detection: {best_config['accuracy']:.1f} objects")
        
        update_config_for_rtx2060(best_config['size'])
    
    return best_config

def create_diamond_test_image(size=1280):
    """Tạo ảnh test với kim cương giả"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img.fill(50)  # Dark background
    
    # Tạo ~20 kim cương giả
    np.random.seed(42)  # Reproducible
    for i in range(20):
        # Random position
        x = np.random.randint(50, size-50)
        y = np.random.randint(50, size-50)
        
        # Random size (1-3mm equivalent)
        radius = np.random.randint(3, 8)
        
        # Draw diamond-like circle
        cv2.circle(img, (x, y), radius, (180, 180, 180), -1)
        cv2.circle(img, (x, y), radius-1, (220, 220, 220), -1)
        
        # Add some sparkle
        cv2.circle(img, (x-1, y-1), 1, (255, 255, 255), -1)
    
    return img

def test_memory_limits():
    """Test giới hạn memory RTX 2060"""
    if not torch.cuda.is_available():
        print("⚠️ GPU không khả dụng - bỏ qua memory test")
        return
        
    model = YOLO('runs/segment/success1/weights/best.pt')
    model.to('cuda')
    
    print("Testing memory with increasing image sizes...")
    
    for size in [1536, 1792, 2048, 2304]:
        try:
            test_img = create_diamond_test_image(size)
            torch.cuda.empty_cache()
            
            mem_before = torch.cuda.memory_allocated()
            results = model(test_img, **YOLO_CONFIG)
            mem_after = torch.cuda.memory_allocated()
            
            mem_used = (mem_after - mem_before) / 1024**3
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ {size}x{size}: {mem_used:.2f}GB used, {total_mem-mem_after/1024**3:.1f}GB free")
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ {size}x{size}: OUT OF MEMORY")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"⚠️ {size}x{size}: Error - {str(e)[:30]}")

def update_config_for_rtx2060(optimal_size):
    """Cập nhật config tự động cho RTX 2060"""
    print(f"\n🔧 Cập nhật config tối ưu cho RTX 2060...")
    
    config_update = f"""
# RTX 2060 Optimal Config (Auto-generated)
YOLO_CONFIG_RTX2060 = {{
    "conf": 0.01,          # Ultra low confidence for 1mm diamonds
    "iou": 0.3,            # IoU threshold
    "max_det": 50000,      # Max detection for RTX 2060
    "augment": False,      # OFF - YOLOv8-seg does not support augmentation
    "agnostic_nms": False, 
    "imgsz": {optimal_size},         # Optimal size for RTX 2060
    "half": True,          # FP16 for RTX 2060
    "verbose": False,
    "retina_masks": True,  # High-quality masks
}}
"""
    
    with open("rtx2060_optimal_config.py", "w", encoding="utf-8") as f:
        f.write(config_update)
    
    print(f"✅ Saved optimal config to rtx2060_optimal_config.py")

if __name__ == "__main__":
    print("🚀 RTX 2060 DIAMOND COUNTER TEST")
    print("=" * 50)
    
    # Main performance test
    best_config = test_rtx2060_performance()
    
    print(f"\n🎉 TEST SUMMARY")
    print("=" * 50)
    if best_config:
        print(f"📊 Best Config: {best_config['size']}x{best_config['size']} ({best_config['desc']})")
        print(f"⚡ FPS: {best_config['fps']:.2f}")
        
        if best_config['fps'] > 3:
            print("✅ RTX 2060 OPTIMIZATION: SUCCESS!")
        else:
            print("⚠️ RTX 2060 OPTIMIZATION: Needs improvement")
    else:
        print("❌ No suitable configuration found")
    
    print("\n💡 Để chạy với config tối ưu:")
    print("   python ap.py")
