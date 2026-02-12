import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# 設定 Matplotlib 使用 Agg 後端，以在背景繪圖而不跳出視窗
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 1. 全域函數：手動計算 Top-k 準確率 (分類專用)
# ==========================================
def calculate_top_k(y_true, y_scores, k=1):
    """
    手動計算 Top-k 分類準確率。
    y_true: 真實標籤 (Numpy 或 Tensor)
    y_scores: 得分矩陣 (Numpy 或 Tensor)
    """
    if torch.is_tensor(y_scores):
        with torch.no_grad():
            n_samples = y_true.size(0)
            num_classes = y_scores.size(1)
            actual_k = min(k, num_classes)
            _, top_k_indices = y_scores.topk(actual_k, dim=1, largest=True, sorted=True)
            y_true_reshaped = y_true.view(-1, 1).expand_as(top_k_indices)
            correct = top_k_indices.eq(y_true_reshaped).sum().item()
            return correct / n_samples
    else:
        # 針對 Random Forest (Numpy) 的處理
        n_samples = len(y_true)
        num_classes = y_scores.shape[1]
        actual_k = min(k, num_classes)
        correct = 0
        for i in range(n_samples):
            # 取得得分最高的前 k 個索引
            top_k_indices = np.argsort(y_scores[i])[-actual_k:][::-1]
            if y_true[i] in top_k_indices:
                correct += 1
        return correct / n_samples

# ==========================================
# 2. 分類模型定義 (基於 PyTorch)
# ==========================================
class PerceptronNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PerceptronNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

class MLPNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # 稍微增加 Dropout 避免過擬合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. 特徵提取與局部裁剪載入邏輯
# ==========================================
def extract_hog_from_crop(img, coords, is_normalized=True):
    h_img, w_img = img.shape[:2]
    if is_normalized:
        xc, yc, w, h = coords
        x1, y1 = int((xc - w/2) * w_img), int((yc - h/2) * h_img)
        x2, y2 = int((xc + w/2) * w_img), int((yc + h/2) * h_img)
    else:
        x1, y1, x2, y2 = [int(v) for v in coords]
    
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
    crop = img[y1:y2, x1:x2]
    
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5: 
        return None
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    hog = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), 
                            _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    return hog.compute(resized).flatten()

def load_deeppcb_classification_data(split_path):
    data, labels = [], []
    img_dir = os.path.join(split_path, "images")
    lbl_dir = os.path.join(split_path, "labels")
    
    if not os.path.exists(img_dir):
        return None, None

    img_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        label_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[4])
                        coords = [float(x) for x in parts[0:4]]
                        is_norm = True if max(coords) <= 1.1 else False
                        feat = extract_hog_from_crop(img, coords, is_normalized=is_norm)
                        if feat is not None:
                            data.append(feat)
                            labels.append(class_id)
        else:
            feat = extract_hog_from_crop(img, [0.5, 0.5, 0.2, 0.2], is_normalized=True)
            if feat is not None:
                data.append(feat)
                labels.append(0)
    return np.array(data), np.array(labels)

# ==========================================
# 4. 訓練流程
# ==========================================
def train_classifier(model, train_loader, val_loader, device, epochs=50):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_correct, total_train = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        history['train_acc'].append(train_correct / (total_train + 1e-7))
        
        model.eval()
        val_correct, total_val = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        history['val_acc'].append(val_correct / (total_val + 1e-7))
    return history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    dataset_root = "../DeepPCB"
    X_train_raw, y_train_raw = load_deeppcb_classification_data(os.path.join(dataset_root, "train"))
    X_val_raw, y_val_raw = load_deeppcb_classification_data(os.path.join(dataset_root, "valid"))
    X_test_raw, y_test_raw = load_deeppcb_classification_data(os.path.join(dataset_root, "test"))
    
    if X_train_raw is None or len(X_train_raw) == 0:
        print("Error: Dataset empty.")
        return

    # 重要：使用 LabelEncoder 統一標籤映射為 [0, 1, 2, ...]
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)
    
    print(f"Label Mapping Check:")
    print(f"  - Classes found: {le.classes_}")
    print(f"  - Train distribution: {Counter(y_train)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train_t = torch.FloatTensor(X_train_scaled)
    X_val_t = torch.FloatTensor(X_val_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)

    final_comparison = []

    # GPU Models
    classifiers = [
        ("Linear Perceptron", PerceptronNet(X_train_t.shape[1], num_classes)),
        ("MLP (Neural Network)", MLPNet(X_train_t.shape[1], num_classes))
    ]

    for name, model in classifiers:
        print(f"\n--- Training: {name} ---")
        history = train_classifier(model, train_loader, val_loader, device)
        
        plt.figure(figsize=(8, 4))
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title(f'{name} - Classification Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(); plt.grid(True)
        plt.savefig(f"{name.lower().replace(' ', '_')}_curve.png"); plt.close()

        model.eval()
        with torch.no_grad():
            out = model(X_test_t.to(device))
            final_comparison.append([name, calculate_top_k(y_test_t.to(device), out, 1), calculate_top_k(y_test_t.to(device), out, 5)])

    # CPU Random Forest (優化參數)
    print("\n--- Training: Random Forest (n_jobs=19) ---")
    # 限制最大深度並調整特徵選取比例，以適應高維度 HOG
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', random_state=42, n_jobs=19)
    rf_model.fit(X_train_scaled, y_train)
    rf_probs = rf_model.predict_proba(X_test_scaled)
    final_comparison.append(["Random Forest", calculate_top_k(y_test, rf_probs, 1), calculate_top_k(y_test, rf_probs, 5)])

    print("\n" + "="*70)
    print(f"{'Classifier Name':<25} | {'Top-1 Accuracy':<15} | {'Top-5 Accuracy':<15}")
    print("-" * 70)
    for res in final_comparison:
        print(f"{res[0]:<25} | {res[1]:<15.4f} | {res[2]:<15.4f}")
    print("="*70)

if __name__ == "__main__":
    main()