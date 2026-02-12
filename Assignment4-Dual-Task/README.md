
# 多任務學習 PCB 瑕疵檢測系統 (Multi-Task PCB Defect Detection)

# Assignment - Multi-Task Learning Analysis Project

課程名稱： 影像處理/深度學習相關課程
學生姓名： 林彥成 Yan-Cheng Lin

資料集：

* Task A (Detection): [DeepPCB Defect Dataset](https://github.com/tangsanli5201/DeepPCB)
* Task B (Segmentation): [VisA (Visual Anomaly) Dataset - PCB Subset](https://github.com/amazon-science/spot-diff)

## 1. 專案簡介 (Introduction)

本專案實作了一個基於 ResNet-18 的多任務學習 (Multi-Task Learning, MTL) 深度神經網路，旨在同時解決兩種性質迥異的 PCB 瑕疵檢測任務：

1. 目標偵測 (Object Detection)：針對 DeepPCB 資料集的二值化線路圖進行瑕疵定位與分類。
2. 語意分割 (Semantic Segmentation)：針對 VisA 資料集的真實彩色照片進行像素級的瑕疵分割。

本研究探討了在「硬參數共享 (Hard Parameter Sharing)」架構下，如何透過雙階段訓練策略 (Two-Stage Training) 與特殊損失函數設計，來平衡不同任務間的梯度競爭與特徵學習。

## 2. 模型架構與實作細節 (Model Architecture)

### 2.1 核心架構：MultiTaskResUNet18

程式碼位於 `pcb_multi_task.py`，採用 PyTorch 框架實作：

* 共享主幹 (Shared Backbone)：
  * 使用 ImageNet 預訓練的 ResNet-18 (移除全連接層)。
  * 負責提取 Layer 1 至 Layer 4 的共用特徵，作為兩個下游任務的基礎。
* 偵測分支 (Task A Head)：
  * 接續 Backbone 的 Layer 4 特徵圖。
  * 透過卷積層將特徵映射為 $8 \times 8$ 的 Grid Detection 輸出。
  * 輸出維度：`(Batch, 1, 8, 8)`，預測該網格是否存在瑕疵中心。
* 分割分支 (Task B Head)：
  * 採用 U-Net Decoder 架構。
  * 透過跳接連接 (Skip Connections) 融合 Backbone 的淺層特徵 (Layer 1-3) 與深層特徵。
  * 逐步上採樣至 $256 \times 256$ 解析度，輸出像素級瑕疵遮罩。

### 2.2 損失函數 (Loss Functions)

* 偵測任務：`BCEWithLogitsLoss`。

* 分割任務：自定義 `FocalDiceLoss`。
  * 結合 Focal Loss (解決前景背景極度不平衡) 與 Dice Loss (優化分割邊界)。
  * 設定 `alpha=0.85`, `gamma=2.0` 以強化對稀疏瑕疵樣本的關注。

### 2.3 訓練策略 (Training Strategy)

為了緩解任務間的負遷移 (Negative Transfer)，本專案採用雙階段訓練法：

1. 預熱階段 (Warmup / SEG-ONLY)：

   * Epoch 1-10。
   * 凍結共享 Backbone 的權重。
   * 僅訓練分割解碼器 (Decoder)，讓分割頭先適應瑕疵特徵，避免初期梯度震盪。

2. 聯合訓練階段 (Joint Training / JOINT)：

   * Epoch 11-25。
   * 解凍 Backbone，同時優化偵測與分割任務。
   * 偵測任務加入訓練，測試其強勢梯度對特徵空間的影響。

3. 資料預處理 (Data Preprocessing)

針對兩個領域差異極大的資料集，實施了以下處理：

* 統一輸入：所有影像 Resize 至 $256 \times 256$ 並進行 ImageNet 標準化。
* Mask 二值化：針對 VisA 分割標籤執行嚴格的閾值處理 (>0)，修正 IoU 計算異常的問題。
* 強制過採樣 (Oversampling)：
  * 由於 VisA 資料集中良品 (無瑕疵) 比例極高，實作了 WeightedRandomSampler。
  * 強制每個 Batch 中包含一定比例的瑕疵樣本，防止模型退化為全背景預測。

## 4. 實驗結果 (Experimental Results)

以下數據基於 RTX 4060 Laptop GPU 環境測得，訓練 25 Epochs 之最終測試集表現：

任務 (Task)	| 指標 (Metric)	| 數值 (Score)	| 備註
| :--- | :--- | :--- | :--- | 
Task A: DeepPCB	| Detection Accuracy	| 99.33%	| 收斂極快，表現優異
Task B: VisA	| Seg IoU (Normal)	| 48.51%	| 背景抑制能力尚可
Task B: VisA	| Seg IoU (Anomaly)	| 23.63%	| 受偵測任務梯度壓抑

#### 結果分析

實驗觀察到顯著的 「梯度支配 (Gradient Domination)」 現象。在進入聯合訓練階段後，特徵簡單明確 (二值化幾何圖形) 的偵測任務迅速主導了 Backbone 的權重更新，導致原本用於處理真實影像紋理的分割特徵被覆寫，使得分割任務的 IoU 在聯合訓練期出現回落。

## 5. 檔案結構與執行說明 (Usage)

### 5.1 專案結構
```
Deep-Learning/
├── pcb_multi_task.py        # 主程式 (模型、資料載入、訓練迴圈)
├── README.md                # 專案說明文件
├── DeepPCB/                 # Task A 資料集路徑
│   ├── train/
│   ├── test/
│   └── ...
└── VisA_PCB_Subset/         # Task B 資料集路徑
    ├── pcb1/
    ├── pcb2/
    └── ...
```

### 5.2 環境需求

* Python 3.8+
* PyTorch, Torchvision
* OpenCV, NumPy, Pandas
* Matplotlib, Scikit-learn

### 5.3 執行方式

直接執行 Python 腳本即可開始訓練與評估：

```
python pcb_multi_task.py
```

訓練完成後，程式將自動生成以下圖表：

* `learning_curves.png`: 訓練過程的 Accuracy 與 IoU 變化曲線。
* `final_test_det_pred.png`: DeepPCB 偵測結果視覺化 (含類別標籤)。
* `final_test_seg_pred.png`: VisA 分割結果視覺化。
* `redraw_*.png`: 強制針對瑕疵樣本的重繪診斷圖。

6. 結論 (Conclusion)

本專案成功建立了一個多任務學習框架，並驗證了 ResNet-Unet 架構在異質 PCB 瑕疵檢測上的可行性。雖然 DeepPCB 的偵測任務達到了近乎完美的準確率，但也揭示了硬參數共享架構在處理領域差異過大 (Domain Gap) 資料時的局限性。未來工作建議採用特徵解耦 (Feature Decoupling) 或 PCGrad 梯度投影技術來改善分割任務的效能。