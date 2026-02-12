# Deep Learning Research Projects: PCB Defect Detection & Framework Analysis
本倉庫彙整了一系列針對印刷電路板（PCB）瑕疵檢測的深度學習研究實作。內容涵蓋從底層 NumPy 運算子開發、主流深度學習框架（PyTorch/TensorFlow）效能基準測試，到進階的多任務學習（Multi-Task Learning）架構設計。

## 核心專案概覽
本研究序列以 DeepPCB 與 VisA 資料集為核心，探討不同演算法架構在工業自動化光學檢測（AOI）場景下的表現。
### 1. 影像分類基礎實作 (Assignment 1)
探討傳統機器學習與基礎神經網路在二值化瑕疵影像上的分類能力。

* 技術重點：Perceptron、MLP 與 Random Forest 之效能對比。
* 成果：建立了自動化局部裁剪（Local Cropping）流程，並透過 Top-1/Top-5 Accuracy 評估模型泛化能力。
  
### 2. 手寫 NumPy LeNet-5 框架 (Assignment 2)
脫離深度學習框架，純手工實作卷積神經網絡，旨在深入理解反向傳播與計算圖邏輯。

* 核心特性：
  * 實作 im2col 與 col2im 演算法，將卷積轉化為矩陣乘法（GEMM）以優化 CPU 運算效率。
  * 手動推導 Swish Activation ($x \cdot \sigma(x)$) 與 Momentum SGD 梯度更新。
*  架構演進：對比了原始 LeNet-5 與採用 $3 \times 3$ 堆疊卷積核的改良版架構。

### 3. 硬體效能基準測試與模型剪枝 (Assignment 3)
針對不同運算後端與執行模式進行壓測，分析工業級場景下的推論延遲與吞吐量。

* 框架對抗：對比 NumPy (CPU)、PyTorch (CUDA)、TensorFlow (Eager) 與 TensorFlow (Static Graph)。
* 關鍵數據：
  * TensorFlow 靜態圖 模式相較於動態圖提升約 1.7 倍 的推論速度。
  * 模型剪枝 (Pruning)：實作 50% 權重剪枝，探討稀疏權重對儲存空間與運算延遲的影響。
  
### 4. 多任務學習 (MTL) 瑕疵檢測系統 (Assignment 4)
開發異構雙任務模型，同時處理目標偵測（Object Detection）與語意分割（Semantic Segmentation）。
架構設計：基於 ResNet-18 Backbone 的硬參數共享（Hard Parameter Sharing）架構。

* 技術挑戰：
  * 設計 FocalDiceLoss 克服瑕疵樣本極度不平衡問題。
  * 分析 梯度支配（Gradient Domination） 現象：探討強勢任務（DeepPCB 偵測）如何壓抑弱勢任務（VisA 分割）的特徵學習。


## 倉庫結構
```
.
├── Assignment1-Image-Classification/  # 基礎分類器與訓練管線
├── Assignment2-LeNet5/                # 純 NumPy 實作與計算圖分析
├── Assignment3-Graph-Compare/         # 框架效能對比與模型剪枝研究
├── Assignment4-Dual-Task/             # 多任務學習與 ResNet-UNet 架構
└── README.md                          # 專案總覽 (本文檔)
```

## 環境需求
* 語言：Python 3.8+
* 框架：PyTorch (Support CUDA), TensorFlow 2.x
* 庫：OpenCV, NumPy, Scikit-learn, Matplotlib
* 硬體推薦：支援 CUDA 之 NVIDIA GPU (實驗環境基於 RTX 4060 Laptop GPU)。
  
## 技術洞察與結論
1. 框架編譯之必要性：在追求高吞吐量的生產環境中，TensorFlow Static Graph 或 TorchScript 的圖編譯技術能顯著降低 Python Runtime 帶來的調度延遲。
2 梯度競爭與負遷移：在多任務學習中，領域差異大（Domain Gap）的資料集易導致特徵覆寫。未來研究建議導入 PCGrad（Projected Conflicting Gradients）或特徵解耦技術。
3. 底層實作價值：透過 NumPy 實作卷積層，能更精準地掌握權重初始化（He Initialization）與激活函數梯度流對深層網路收斂的影響。

---

Author: Yan-Cheng Lin (林彥成)
Academic Context: Imaging Processing & Deep Learning Research Series
