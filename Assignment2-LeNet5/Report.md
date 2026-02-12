# **DeepPCB 缺陷分類任務：LeNet-5 與改良型架構之性能研究**

# **Assignment 2 - LeNet5 Report**

課程名稱： 影像處理/深度學習相關課程
學生姓名： 林彥成 Yan-Cheng Lin
資料集： DeepPCB Defect Dataset (https://www.kaggle.com/datasets/arnablaha05/deep-pcb)
## **1. 完整原始碼連結 (Complete Source Codes)**

* **Colab / GitHub Link:** https://github.com/cvyancheng/Deep-Learning
* **主要檔案：** `lenet_mlp_numpy.py` 

## **2. 實驗背景與模型架構**

本研究基於純 NumPy 環境實作了兩套卷積神經網絡（CNN）架構，用於處理 DeepPCB 印刷電路板缺陷裁切影像的分類任務（共 7 個類別，包含背景與 6 種缺陷）。

### **A. Naive LeNet-5 (基準模型)**

* **卷積層 (C1/C3)**：採用傳統 ![][image1] 卷積核。  
* **激活函數**：使用 ReLU。  
* **下採樣 (S2/S4)**：$2 \times 2$ 最大池化（Max Pooling）。  
* **全連接層**：包含兩層隱藏層（120, 84），輸出層對應類別數。

### **B. Improved LeNet-5 (改良型模型)**

* **卷積核優化**：將 $5 \times 5$ 核替換為連續的 $3 \times 3$ 卷積核堆疊。此設計能在維持感受野的同時，增加非線性層數並減少參數負擔。  
* **深度增加**：額外增加一個卷積層（C2），強化對 PCB 細微特徵（如短路、開路）的提取能力。  
* **激活函數 (Swish)**：採用 Swish 函數（$x \cdot \text{sigmoid}(x)$），其平滑且非單調的特性有效緩解了神經元壞死問題，優化了深層網絡的梯度流。  
* **寬度增加**：Filter 數量從 Naive 版的 (6, 16) 提升至 (16, 16, 32)。

## **3. 訓練策略與梯度改進**

為了在不使用 PyTorch/TensorFlow 的情況下實現穩定訓練，本實作導入了以下關鍵策略：

* **梯度流修正 (Gradient Flow Improvement)**：實作了精確的 `col2im`邏輯與 MaxPool Mask，確保損失訊號能精確傳回輸入層。  
* **動量優化 (Momentum SGD)**：在權重更新中加入動量（$\mu = 0.9$），利用歷史梯度緩衝（Velocity Buffers）衝過損失表面的平坦區域，顯著提升收斂速度與穩定性。  
* **數據標準化**：對輸入影像進行全域標準化（Zero-mean, Unit-variance），確保卷積核在初始化階段能接收到穩定的輸入分佈。

## **4. 實驗結果分析**

### **A. 準確率曲線觀察 (Accuracy Curves)**

根據實驗記錄的 `accuracy_comparison.png`：
1. **收斂速度**：兩個版本均在 10 個 Epoch 內迅速突破 98% 的訓練準確率。這歸功於 Momentum SGD 的加速作用。  
2. **泛化表現**：Naive 版在 Epoch 15 後驗證準確率趨於飽和（約 0.9801）；Improved 版雖然收斂初期略慢於 Naive，但在後續展現出更好的穩定性，最高驗證準確率達到 **0.9821**。

![accuracy_comparison.png](accuracy_comparison.png)

### **B. 驗證集預測結果 (Validation Top-1 Accuracy)**

* **Naive LeNet5**：最終驗證準確率穩定在 **0.9791**。  
* **Improved LeNet5**：最終驗證準確率穩定在 **0.9811**。  
* **分析**：Improved 版透過更深的卷積層提取到了更精細的瑕疵特徵，對於邊界模糊的缺陷（如 Mousebite）辨識度更高。

### **C. 測試集評估 (Testing Set Results)**

模型在未見過的測試集上表現極佳，顯示模型具備高度泛化能力：

* **Naive LeNet5 (Top-1)**：**0.9797**  
* **Improved LeNet5 (Top-1)**：**0.9807**

## **5. 結論**

在本項研究中，Improved LeNet5 在所有指標上均微幅領先 Naive 版本。雖然 $3 \times 3$ 卷積與 Swish 函數增加了計算開銷，但其帶來的特徵表徵能力與梯度流穩定性，在 DeepPCB 資料集上取得了更優異的 Top-1 準確率。本次實驗證明了在純 NumPy 框架下，透過合理的梯度歷史（Momentum）與層級設計優化，可以構建出媲美現代深度學習框架的工業級影像分類器。