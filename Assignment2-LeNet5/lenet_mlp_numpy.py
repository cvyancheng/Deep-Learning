import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 設定 Matplotlib 使用 Agg 後端
import matplotlib
matplotlib.use('Agg')

# --- 1. 高效卷積工具函式 (純 NumPy 實現 BP 關鍵) ---

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """將矩陣梯度還原回影像張量空間"""
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """將影像塊展開為行，以便進行高效矩陣乘法"""
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

# --- 2. 基礎組件與激活函數 ---

class Layer:
    def __init__(self): self.input = None
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Swish(Layer):
    """Improved LeNet5 指定激活函數: x * sigmoid(x)"""
    def forward(self, input):
        self.input = input
        self.sig = 1 / (1 + np.exp(-np.clip(input, -500, 500)))
        return input * self.sig
    def backward(self, output_gradient, learning_rate):
        # 導數: f'(x) = f(x) + sigmoid(x)(1 - f(x))
        grad = self.sig + (self.input * self.sig * (1 - self.sig))
        return output_gradient * grad

class Softmax:
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exps / (np.sum(exps, axis=1, keepdims=True) + 1e-8)

# --- 3. 帶有 Momentum SGD 的核心層級 ---

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        # He 初始化
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias
    def backward(self, output_gradient, learning_rate, momentum=0.9):
        dw = np.dot(self.input.T, output_gradient)
        db = np.sum(output_gradient, axis=0, keepdims=True)
        dx = np.dot(output_gradient, self.weights.T)
        # Momentum SGD 更新
        self.v_w = momentum * self.v_w - learning_rate * dw
        self.v_b = momentum * self.v_b - learning_rate * db
        self.weights += self.v_w
        self.bias += self.v_b
        return dx

class ConvLayer(Layer):
    def __init__(self, input_channels, n_filters, filter_size, stride=1, padding=0):
        self.n_filters, self.filter_size, self.stride, self.padding = n_filters, filter_size, stride, padding
        self.input_channels = input_channels
        limit = np.sqrt(2./(input_channels * filter_size**2))
        self.weights = np.random.randn(n_filters, input_channels, filter_size, filter_size) * limit
        self.bias = np.zeros(n_filters)
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)
    def forward(self, input_data):
        self.input = input_data
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.filter_size) // self.stride + 1
        self.col = im2col(input_data, self.filter_size, self.filter_size, self.stride, self.padding)
        self.col_W = self.weights.reshape(self.n_filters, -1).T
        return (np.dot(self.col, self.col_W) + self.bias).reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    def backward(self, output_gradient, learning_rate, momentum=0.9):
        dout = output_gradient.transpose(0, 2, 3, 1).reshape(-1, self.n_filters)
        db = np.sum(dout, axis=0)
        dw = np.dot(self.col.T, dout).transpose(1, 0).reshape(self.weights.shape)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.input.shape, self.filter_size, self.filter_size, self.stride, self.padding)
        # Momentum SGD 更新
        self.v_w = momentum * self.v_w - learning_rate * dw
        self.v_b = momentum * self.v_b - learning_rate * db
        self.weights += self.v_w
        self.bias += self.v_b
        return dx

class MaxPool(Layer):
    def __init__(self, size=2, stride=2): self.size, self.stride = size, stride
    def forward(self, input_data):
        self.input = input_data
        N, C, H, W = input_data.shape
        out_h, out_w = H // self.stride, W // self.stride
        view = input_data.reshape(N, C, out_h, self.stride, out_w, self.stride)
        out = view.max(axis=(3, 5))
        self.arg_max = (view == out[:, :, :, np.newaxis, :, np.newaxis])
        return out
    def backward(self, output_gradient, learning_rate):
        N, C, out_h, out_w = output_gradient.shape
        dview = np.zeros((N, C, out_h, self.size, out_w, self.size))
        dview += output_gradient[:, :, :, np.newaxis, :, np.newaxis] * self.arg_max
        return dview.reshape(self.input.shape)

# --- 4. 根據一對一命名邏輯 (X.jpg -> X.txt) 優化資料載入 ---

def load_deeppcb_data_refined(split_path, target_size=(32, 32)):
    data, raw_labels = [], []
    img_dir = os.path.join(split_path, "images")
    lbl_dir = os.path.join(split_path, "labels")
    if not os.path.exists(img_dir): return None, None
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"掃描 {split_path}: 發現 {len(img_files)} 張影像...")

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h_i, w_i = img.shape[:2]
        
        # 精確匹配邏輯: 01_PCB__1.jpg -> 01_PCB__1.txt
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                if not lines: # 無瑕疵則作為背景 (Class 0)
                    crop = img[h_i//2-16:h_i//2+16, w_i//2-16:w_i//2+16]
                    data.append(cv2.resize(crop, target_size).transpose(2,0,1)/255.0)
                    raw_labels.append(0)
                else:
                    for line in lines:
                        p = line.split()
                        if len(p) == 5:
                            xc, yc, w, h = map(float, p[0:4])
                            # 自動判定座標是否已歸一化
                            is_norm = (max(xc, yc, w, h) <= 1.1)
                            if is_norm:
                                x1, y1 = int((xc-w/2)*w_i), int((yc-h/2)*h_i)
                                x2, y2 = int((xc+w/2)*w_i), int((yc+h/2)*h_i)
                            else:
                                x1, y1, x2, y2 = map(int, p[0:4])
                            
                            crop = img[max(0,y1):min(h_i,y2), max(0,x1):min(w_i,x2)]
                            if crop.size == 0: continue
                            data.append(cv2.resize(crop, target_size).transpose(2,0,1)/255.0)
                            raw_labels.append(int(p[4]))
        else: # 若無對應標籤檔，依原腳本邏輯取中心作為背景
            crop = img[h_i//2-16:h_i//2+16, w_i//2-16:w_i//2+16]
            data.append(cv2.resize(crop, target_size).transpose(2,0,1)/255.0)
            raw_labels.append(0)
    
    if len(data) == 0: return None, None
    X = np.array(data)
    # 進行特徵縮放與標準化 (提高收斂穩定性)
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    y = np.eye(7)[np.array(raw_labels, dtype=int)] # 固定 7 類 (0-6)
    return X, y

# --- 5. 模型架構定義 ---

class NaiveLeNet5:
    """任務 2: 傳統 LeNet5 (5x5, ReLU)"""
    def __init__(self, n_classes):
        self.conv = [ConvLayer(3, 6, 5), ReLU(), MaxPool(2, 2), ConvLayer(6, 16, 5), ReLU(), MaxPool(2, 2)]
        self.fc = [FCLayer(16*5*5, 120), ReLU(), FCLayer(120, 84), ReLU(), FCLayer(84, n_classes)]
    def forward(self, x):
        for l in self.conv: x = l.forward(x)
        self.flat_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        for l in self.fc: x = l.forward(x)
        return x
    def train_step(self, x, y, lr):
        logits = self.forward(x); probs = Softmax().forward(logits); grad = (probs - y)/len(x)
        for l in reversed(self.fc): grad = l.backward(grad, lr)
        grad = grad.reshape(self.flat_shape)
        for l in reversed(self.conv): grad = l.backward(grad, lr)
        return probs

class ImprovedLeNet5:
    """任務 3: 改進版 LeNet5 (3x3, Swish, +1 Conv)"""
    def __init__(self, n_classes):
        self.conv = [ConvLayer(3, 16, 3, padding=1), Swish(), 
                     ConvLayer(16, 16, 3, padding=1), Swish(), MaxPool(2, 2),
                     ConvLayer(16, 32, 3, padding=1), Swish(), MaxPool(2, 2)]
        self.fc = [FCLayer(32*8*8, 120), Swish(), FCLayer(120, n_classes)]
    def forward(self, x):
        for l in self.conv: x = l.forward(x)
        self.flat_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        for l in self.fc: x = l.forward(x)
        return x
    def train_step(self, x, y, lr):
        logits = self.forward(x); probs = Softmax().forward(logits); grad = (probs - y)/len(x)
        for l in reversed(self.fc): grad = l.backward(grad, lr)
        grad = grad.reshape(self.flat_shape)
        for l in reversed(self.conv): grad = l.backward(grad, lr)
        return probs

def evaluate(model, x, y):
    if x is None or len(x) == 0: return 0.0
    batch_size = 64
    correct = 0
    for i in range(0, len(x), batch_size):
        bx, by = x[i:i+batch_size], y[i:i+batch_size]
        probs = Softmax().forward(model.forward(bx))
        correct += np.sum(np.argmax(probs, axis=1) == np.argmax(by, axis=1))
    return correct / len(x)

if __name__ == "__main__":
    base_path = "../DeepPCB"
    print("--- 執行全面資料載入 ---")
    train_x, train_y = load_deeppcb_data_refined(os.path.join(base_path, "train"))
    val_x, val_y = load_deeppcb_data_refined(os.path.join(base_path, "valid"))
    test_x, test_y = load_deeppcb_data_refined(os.path.join(base_path, "test"))
    
    if train_x is not None:
        print(f"資料載入成功。總訓練樣本: {len(train_x)} | 驗證: {len(val_x)} | 測試: {len(test_x)}")
        batch_size, epochs, lr = 64, 30, 0.005 # 調低 lr 配合 Momentum
        all_histories = {}
        
        for ModelClass, name in [(NaiveLeNet5, "Naive"), (ImprovedLeNet5, "Improved")]:
            model = ModelClass(7)
            history = {"train_acc": [], "val_acc": []}
            print(f"\n--- 開始訓練 {name} (Momentum SGD) ---")
            for epoch in range(epochs):
                idx = np.random.permutation(len(train_x))
                train_x, train_y = train_x[idx], train_y[idx]
                for i in range(0, len(train_x), batch_size):
                    model.train_step(train_x[i:i+batch_size], train_y[i:i+batch_size], lr)
                
                t_acc, v_acc = evaluate(model, train_x, train_y), evaluate(model, val_x, val_y)
                history["train_acc"].append(t_acc); history["val_acc"].append(v_acc)
                if epoch % 5 == 0:
                    print(f"Epoch {epoch:02d} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
            
            test_acc = evaluate(model, test_x, test_y)
            print(f"--- {name} 測試集 Top-1 準確率: {test_acc:.4f} ---")
            all_histories[name] = history

        # 繪圖分析研究
        plt.figure(figsize=(10, 6))
        for name, hist in all_histories.items():
            plt.plot(hist["train_acc"], label=f"{name} Train")
            plt.plot(hist["val_acc"], linestyle='--', label=f"{name} Val")
        plt.title("Performance Analysis (Naive vs Improved LeNet5)"); plt.xlabel("Epochs"); plt.ylabel("Accuracy")
        plt.legend(); plt.grid(True); plt.savefig("accuracy_comparison.png")
    else:
        print("致命錯誤：未能從路徑載入任何訓練數據。")