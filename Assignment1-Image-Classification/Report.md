# **DeepPCB 缺陷分類任務**
# **Assignment 1 - Image Classification Report**

課程名稱： 影像處理/深度學習相關課程
學生姓名： 林彥成 Yan-Cheng Lin
資料集： DeepPCB Defect Dataset (https://www.kaggle.com/datasets/arnablaha05/deep-pcb)

## **1. 完整原始碼連結 (Complete Source Codes)**

* **Colab / GitHub Link:** https://github.com/cvyancheng/Deep-Learning
* **主要檔案：** `image_classification_gpu.py` (包含完整 Pipeline)

## **2. 使用模型說明 (Models Used)**

本作業共採用三種分類器進行性能評估，涵蓋線性模型、非線性深度學習模型以及整合學習模型：

1. **Linear Perceptron (線性感知器):** * 作為 Baseline 模型。  
   * 實作方式：使用 PyTorch 的 `nn.Linear` 層，配合 `CrossEntropyLoss` 進行隨機梯度下降。  
2. **MLP (多層感知機):** \* 非線性深度學習代表。  
   * 實作方式：三層全連接層結構（512-256-Output），加入 ReLU 激活函數與 Dropout (0.3) 以防止過擬合。  
3. **Random Forest (整合學習 \- Ensemble Learning):**  
   * 基於 Bagging 策略的整合學習分類器。  
   * 實作方式：使用 `RandomForestClassifier`，配置 100 棵決策樹，並透過多核心並行運算 (`n_jobs=19`) 加速處理高維 HOG 特徵。

## **3. 影像特徵提取與預處理 (Feature Extraction & Preprocessing)**

### **3.1 瑕疵局部裁剪 (Local Cropping)**

由於 DeepPCB 的原始影像中瑕疵佔比極小，直接進行全域縮放會導致特徵被背景電路板雜訊淹沒。本實作根據 `labels` 資料夾中的 YOLO 標籤格式（`x1 y1 x2 y2 type`），將瑕疵區域精確裁剪出來，確保分類器學習到的是瑕疵本身的形態特徵。

### **3.2 HOG 特徵提取**

* **方法：** Histogram of Oriented Gradients (HOG)。  
* **規格：** 將裁剪後的影像統一縮放為 $64 \times 64$，提取梯度方向直方圖。  
* **目的：** HOG 對物體的幾何與光學形變具有良好的鲁棒性，適合描述 PCB 上的斷路、短路等邊緣特徵。

## **4. 訓練與驗證準確率曲線 (Learning Curves)**

透過 50 個 Epoch 的訓練，模型收斂情況如下：

### **4.1 Linear Perceptron 學習曲線**

![lp_learning_curve](linear_perceptron_curve.png)

* **分析：** 線性模型收斂較快，但訓練與驗證準確率之間存在明顯差距，說明模型容量（Capacity）有限，難以完美擬合複雜的瑕疵特徵。

### **4.2 MLP (Neural Network) 學習曲線**

![mlp_learning_curve](mlp_(neural_network)_curve.png)

* **分析：** MLP 在訓練集上展現了極強的擬合能力（接近 98%），驗證準確率也穩定上升至 75% 左右。雖然存在輕微過擬合，但整體表現最優。

## **5. 測試集預測結果 (Performance Comparison)**

本報告之 **Top-1** 與 **Top-5** 準確率均由手寫函數 `calculate_top_k` 計算，**未調用任何現成評估工具箱**（如 `sklearn.metrics`），確保指標計算的透明度與準確性。

| 分類器模型 (Classifier Name) | Top-1 Accuracy | Top-5 Accuracy |
| :---- | :---- | :---- |
| **Linear Perceptron** | 0.6047 | 0.9959 |
| **MLP (Neural Network)** | **0.7541** | **0.9970** |
| **Random Forest (Ensemble)** | 0.6972 | 0.9959 |

### **性能分析與對比：**

* **MLP 表現最佳：** 瑕疵特徵在 HOG 空間中並非線性可分。MLP 透過隱藏層的非線性轉換，能更精準地捕捉瑕疵間的微小差異。  
* **整合學習優勢：** Random Forest 的表現優於線性感知器約 9%，證明了透過多棵決策樹整合後的分類邊界比單一超平面更具泛化能力。  
* **Top-5 接近 100%：** 說明即使 Top-1 預測錯誤，正確答案也幾乎都落在模型預測的前五個候選類別中，反映出模型已初步掌握各瑕疵類別的特徵規律。