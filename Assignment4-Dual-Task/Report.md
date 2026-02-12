# 多任務 PCB 瑕疵檢測模型訓練診斷報告

# Assignment 4 - Multi-Task Learning Analysis Report

課程名稱： 影像處理/深度學習相關課程
學生姓名： 林彥成 Yan-Cheng Lin
資料集： DeepPCB Defect Dataset (Task A) [https://www.kaggle.com/datasets/arnablaha05/deep-pcb] & VisA PCB Subset (Task B) [https://github.com/amazon-science/spot-diff]

## 1. 前言 (Introduction)

本實驗旨在探討「多任務學習（Multi-Task Learning, MTL）」架構在印刷電路板（PCB）瑕疵檢測中的應用與挑戰。實驗整合了兩個性質迥異的任務：針對 DeepPCB 資料集的目標偵測（Object Detection）與針對 VisA 資料集的語意分割（Semantic Segmentation）。

透過共用卷積神經網路（CNN）提取特徵，我們期望模型能學習到更泛化的瑕疵表徵。然而，由於兩組資料集在影像域（Domain）上的巨大差異（二值化簡圖 vs. 真實彩色照片），以及任務本身優化難度的不對稱性，本報告將深入剖析訓練過程中出現的「梯度支配（Gradient Domination）」與「負遷移（Negative Transfer）」現象，並提出架構上的改進建議。

![DeepPCB 資料集](gallery_deeppcb.png)
DeepPCB 資料集


![VisA 資料集](gallery_visa.png)
VisA 資料集

## 2. 完整原始碼連結 (Complete Source Codes)

GitHub Link: https://github.com/cvyancheng/Deep-Learning

主要檔案說明：

* `pcb_multi_task.py`：核心實驗代碼。包含 DeepPCB 與 VisA 資料集的預處理、MultiTaskResUNet18 模型架構定義、FocalDiceLoss 損失函數實作，以及雙階段（Two-Stage）訓練流程控制。

## 3. 使用模型說明 (Model Implementation)

### 3.1 模型架構：MultiTaskResUNet18

* 本實驗採用「硬參數共享（Hard Parameter Sharing）」策略，建構了一個基於 ResNet-18 的雙頭網路：
* 共享編碼器 (Shared Encoder)：使用預訓練的 ResNet-18（移除全連接層）作為主幹網路（Backbone），負責提取影像的共用特徵（Layer 1~4）。
* 偵測解碼頭 (Detection Head)：接續 Layer 4 特徵圖，透過 $3 \times 3$ 與 $1 \times 1$ 卷積層輸出 $8 \times 8$ 的 Grid Detection 結果，負責 DeepPCB 的瑕疵定位。
* 分割解碼頭 (Segmentation Head)：採用 U-Net 架構的解碼器設計，將 Layer 4 特徵透過跳接連接（Skip Connections）與淺層特徵（Layer 1~3）融合，逐步上採樣至 $256 \times 256$ 解析度，負責 VisA 的像素級瑕疵分割。

### 3.2 損失函數與訓練策略

* 損失函數：
  * 偵測任務：BCEWithLogitsLoss。
  * 分割任務：FocalDiceLoss（結合 Focal Loss 解決類別不平衡與 Dice Loss 優化邊界）。
* 雙階段訓練：
  * Phase 1 (Epoch 1-10)：凍結 Backbone，僅訓練分割解碼器（Segmentation Warmup）。
  * Phase 2 (Epoch 11-25)：解凍全網，聯合訓練（Joint Training）。

## 4. 影像特徵提取與預處理 (Feature Extraction & Preprocessing)

本實驗面臨的最大挑戰在於輸入資料的極端領域差異：

* DeepPCB (Task A)：影像呈現高對比、近乎二值化的線路佈局，特徵具備高度幾何規律性，背景純淨。
* VisA (Task B)：影像為真實拍攝的 PCBA 彩色照片，包含複雜的光影變化、焊接反光與非瑕疵的電子元件紋理。

預處理流程：

1. 尺寸統一：所有輸入影像均調整為 $256 \times 256$。
2. 標準化：使用 ImageNet 統計量（Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]）進行正規化。
3. Mask 二值化：針對 VisA 分割標籤，強制執行閾值處理，確保 Ground Truth 為純粹的 0/1 分佈。
4. 強制過採樣 (Oversampling)：針對 VisA 資料集引入 WeightedRandomSampler，解決良品與瑕疵樣本極度不平衡的問題，防止模型退化為全背景預測。

## 5. 學習曲線 (Learning Curves)

![學習曲線](learning_curves.png)
上圖展示了雙階段訓練的動態變化，清楚反映了任務間的交互作用：

* 預熱階段 (Epoch 1-10)：
在此階段，Backbone 被凍結。Seg IoU(Anom)（紅色曲線）穩定上升至 0.35，顯示分割頭能有效學習瑕疵特徵。

* 聯合訓練階段 (Epoch 11-25)：
進入 Epoch 11 解凍後，Det Acc（藍色曲線）瞬間從 0.12 飆升至 0.99 並維持高檔。然而，這伴隨著 Seg IoU（綠色與紅色曲線）的劇烈震盪與回落。這證實了偵測任務的強梯度主導了共享特徵空間的更新，迫使 Backbone 犧牲了有利於分割的高頻紋理特徵。

## 6. 測試集預測結果 (Test Set Prediction Results)

### 6.1 定量評估

在最終測試集上，模型表現呈現顯著的任務效能不對稱：

* Task A (DeepPCB 偵測)：
  * Accuracy: **0.9933** (極優)
* Task B (VisA 分割)：
  * IoU (Normal/Background): 0.4851
  * IoU (Anomaly/Defect): **0.2363**

### 6.2 定性視覺化分析

![DeepPCB 偵測結果](redraw_det_pred_defects.png)
DeepPCB 偵測結果：
模型展現了近乎完美的定位能力，藍色邊界框精準鎖定了各類瑕疵（Open, Mousebite, Short 等），且無誤報。

![VisA 分割結果](redraw_seg_pred_defects.png)
VisA 分割結果：
相較於訓練初期的完全發散，模型已能大致抑制背景雜訊（Norm IoU 0.48），但在真實瑕疵的輪廓描繪上顯得粗糙且不連續（Anom IoU 0.23），顯示底層空間解析能力在聯合訓練中受到壓抑。

## 7. 結論 (Conclusion)

本實驗透過實作多任務 ResNet18-UNet 架構，成功驗證了多任務學習在異質資料集上的行為特徵。

主要發現：

1. 梯度支配 (Gradient Domination)：目標偵測任務（DeepPCB）由於特徵簡單（二值幾何），其優化速度遠快於語意分割任務，導致共享參數被「劫持」。
2. 負遷移 (Negative Transfer)：聯合訓練並未帶來預期的協同效應，反而導致分割任務的效能低於單獨訓練（Warmup）階段的峰值。
3. 架構限制：硬參數共享（Hard-sharing）對於領域差異過大（Domain Gap）的任務組合並不適用。

未來改進建議：
為突破目前的次優解瓶頸，建議採用 **特徵解耦(Feature Decoupling)** 策略，例如將 ResNet 的淺層（Stem/Layer1）分岔訓練，僅在深層共享語意；或引入 PCGrad (Projecting Conflicting Gradients) 演算法，將衝突的梯度投影至法平面，以數學手段消除任務間的破壞性干擾。