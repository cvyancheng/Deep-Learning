import os
import time
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

# 解決 Keras 3 與 tf_keras 衝突
try:
    import tf_keras as keras
    from tf_keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

try:
    import tensorflow_model_optimization as tfmot
    HAS_TF_OPTIM = True
except ImportError:
    HAS_TF_OPTIM = False

# --- 1. 資料載入邏輯 (僅載入影像進行推論測試) ---

def load_benchmark_data(base_path, split="test", target_size=(32, 32)):
    """載入影像數據用於基準測試"""
    data = []
    split_path = os.path.join(base_path, split)
    img_dir = os.path.join(split_path, "images")
    
    if not os.path.exists(img_dir):
        print(f"警告：找不到路徑 {img_dir}，將使用 1000 筆模擬數據。")
        return np.random.randn(1000, 3, 32, 32).astype(np.float32)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img_resized = cv2.resize(img, target_size)
        # NCHW 格式 (PyTorch/NumPy 預設)
        data.append(img_resized.transpose(2, 0, 1) / 255.0)
        
    X = np.array(data).astype(np.float32)
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    return X

# --- 2. 模型架構定義 ---

class LeNet5Pytorch(nn.Module):
    def __init__(self, n_classes=7):
        super(LeNet5Pytorch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_tf_lenet(n_classes=7):
    return keras.Sequential([
        layers.Conv2D(6, 5, activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(16, 5, activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

# --- 3. 效能基準測試核心 ---

def measure_hardware_performance(predict_fn, input_data, iterations=10):
    """
    測量硬體效能
    iterations: 重複執行次數以取得穩定平均值
    """
    # Warm up: 排除冷啟動影響
    predict_fn(input_data[:min(5, len(input_data))])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        predict_fn(input_data)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    total_elapsed = time.time() - start_time
    avg_latency_per_img = (total_elapsed / iterations) / len(input_data)
    throughput_fps = 1.0 / avg_latency_per_img
    
    return avg_latency_per_img * 1000, throughput_fps # 回傳 ms 與 fps

# --- 4. 主執行邏輯 ---

def run_benchmarks():
    base_path = "../DeepPCB"
    print("正在準備基準測試數據...")
    X_test_pt = load_benchmark_data(base_path, "test")
    X_test_tf = X_test_pt.transpose(0, 2, 3, 1) # NHWC
    
    n_samples = len(X_test_pt)
    n_classes = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    print(f"測試樣本數: {n_samples} | 運行設備: {device}")

    # --- (A) PyTorch (CUDA/CPU) ---
    pt_model = LeNet5Pytorch(n_classes).to(device)
    pt_model.eval()
    def pt_fn(x):
        with torch.no_grad():
            return pt_model(torch.from_numpy(x).to(device))
    
    latency, fps = measure_hardware_performance(pt_fn, X_test_pt)
    results['PyTorch'] = {'latency': latency, 'fps': fps, 'params': sum(p.numel() for p in pt_model.parameters())}

    # --- (B) TensorFlow Dynamic (Eager) ---
    tf_model = create_tf_lenet(n_classes)
    def tf_dyn_fn(x): return tf_model(x, training=False)
    
    latency, fps = measure_hardware_performance(tf_dyn_fn, X_test_tf)
    results['TF (Dynamic)'] = {'latency': latency, 'fps': fps, 'params': tf_model.count_params()}

    # --- (C) TensorFlow Static (Graph Mode) ---
    @tf.function(reduce_retracing=True)
    def tf_static_fn(x): return tf_model(x, training=False)
    tf_static_fn(tf.constant(X_test_tf[:1])) # 預編譯
    
    latency, fps = measure_hardware_performance(lambda x: tf_static_fn(tf.constant(x)), X_test_tf)
    results['TF (Static)'] = {'latency': latency, 'fps': fps, 'params': tf_model.count_params()}

    # --- (D) NumPy (手寫版本) ---
    try:
        from lenet_mlp_numpy import NaiveLeNet5
        np_model = NaiveLeNet5(n_classes)
        latency, fps = measure_hardware_performance(lambda x: np_model.forward(x), X_test_pt, iterations=1)
        results['Handcrafted (NumPy)'] = {'latency': latency, 'fps': fps, 'params': "~62,000"}
    except Exception:
        pass

    # --- (E) TF Pruned 50% ---
    if HAS_TF_OPTIM:
        try:
            p_model = tfmot.sparsity.keras.strip_pruning(
                tfmot.sparsity.keras.prune_low_magnitude(create_tf_lenet(n_classes), 
                tfmot.sparsity.keras.ConstantSparsity(0.5, 0)))
            @tf.function(reduce_retracing=True)
            def prune_fn(x): return p_model(x, training=False)
            
            latency, fps = measure_hardware_performance(lambda x: prune_inf(tf.constant(x)), X_test_tf)
            results['TF (Pruned 50%)'] = {'latency': latency, 'fps': fps, 'params': p_model.count_params()}
        except: pass

    # --- 輸出報表 ---
    print("\n" + "="*90)
    print(f"{'架構版本 (Framework)':<22} | {'參數數量':<12} | {'延遲 (ms/img)':<15} | {'吞吐量 (FPS)':<12}")
    print("-" * 90)
    
    for name, data in results.items():
        print(f"{name:<22} | {str(data['params']):<12} | {data['latency']:<15.4f} | {data['fps']:<12.2f}")
    print("="*90)

if __name__ == "__main__":
    run_benchmarks()