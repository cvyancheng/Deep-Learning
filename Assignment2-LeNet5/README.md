# **LeNet-5 Implementation from Scratch using NumPy**

本專案是一個基於純 **NumPy** 實作的卷積神經網絡（CNN）框架，重點展示了計算圖（Computational Graph）的手動建構與反向傳播（Back-propagation）演算法。專案針對 **DeepPCB** 印刷電路板缺陷資料集進行了優化，並實作了原始 LeNet-5 與改良型架構的效能對比。

## **1. 核心特性**

* **純 NumPy 實作**：除了影像處理（OpenCV）與資料視覺化（Matplotlib）外，所有神經網絡層級（卷積、池化、全連接）及其優化演算法均不使用任何深度學習框架（如 PyTorch 或 TensorFlow）。  
* **高效卷積運算**：實作了 `im2col` 與 `col2im` 演算法，將卷積運算轉化為通用矩陣乘法（GEMM），顯著提升在 CPU 環境下的運算效率。  
* **計算圖反向傳播**：  
  * **ConvLayer**：手動推導並實作權重、偏置與輸入張量的梯度計算。  
  * **MaxPool**：透過記錄 `arg_max` 索引實現梯度的精確回傳。  
  * **Swish Activation**：實作了改良型 LeNet 指定的 $x \cdot \text{sigmoid}(x)$ 激活函數及其導數。  
* **優化演算法**：導入了 **Momentum SGD**（帶動量的隨機梯度下降），有效加速收斂並減少震盪。

## **2. 系統架構**

### **模型版本**

1. **Naive LeNet-5**：  
   * $5 \times 5$ 卷積核。  
   * ReLU 激活函數。  
   * 傳統的 C1-S2-C3-S4-FC 結構。  
2. **Improved LeNet-5**：  
   * $3 \times 3$ 卷積核堆疊。  
   * Swish 激活函數（$x \cdot \text{sigmoid}(x)$）。  
   * 額外的卷積層（C2）與更多的通道數（Filter count）。

### **資料預處理**

* **DeepPCB 匹配邏輯**：針對 `01_PCB__1.jpg` 與 `01_PCB__1.txt` 進行一對一精確匹配。  
* **局部裁剪 (Local Cropping)**：根據標籤中的標註座標自動裁剪缺陷區域，並支援歸一化與像素座標的自動判定。  
* **特徵標準化**：對輸入張量進行 Zero-mean 與 Unit-variance 處理。

## **3. 環境需求**

* Python 3.x  
* NumPy  
* OpenCV (opencv-python)  
* Matplotlib

## **4. 使用方法**

請確保您的資料集路徑結構如下：

```
../DeepPCB/  
├── train/  
│   ├── images/  
│   └── labels/  
├── valid/  
│   └── ...  
└── test/  
    └── ...
```
執行訓練與評估：

```
python lenet\_mlp\_numpy.py
```

## **5. 效能分析**

訓練完成後，程式會自動生成 `accuracy_comparison.png`，展示兩個模型在訓練集與驗證集上的收斂曲線，並在終端機輸出測試集的 **Top-1 Accuracy**。

### **實驗數據參考 (DeepPCB)**

* **訓練集**：8,024 樣本  
* **驗證集**：1,005 樣本  
* **測試集**：984 樣本  
* **平均 Top-1 準確率**：約 98%

## **6. 技術洞察**

在純 NumPy 環境下，**Momentum SGD** 的引入是模型能從隨機猜測水平躍升至高精確度的關鍵。此外，透過精確的 `col2im` 還原梯度，解決了深層網路中梯度流斷裂的問題，使得 Improved 版的架構優勢得以體現。