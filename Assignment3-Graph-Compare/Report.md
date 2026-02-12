
# **LeNet55 框架硬體效能基準測試**
# **Assignment 3 - Assignment3-Graph-Compare Report**

課程名稱： 影像處理/深度學習相關課程
學生姓名： 林彥成 Yan-Cheng Lin
資料集： DeepPCB Defect Dataset (https://www.kaggle.com/datasets/arnablaha05/deep-pcb)

## 1. 完整原始碼連結 (Complete Source Codes)

GitHub Link: https://github.com/cvyancheng/Deep-Learning

主要檔案說明：

* `lenet_mlp_numpy.py`：純 NumPy 實作版本，包含模型訓練、驗證與測試集評估邏輯。
* `lenet_framework_comparison.py`：框架效能對比工具，實作了 PyTorch、TensorFlow (Eager/Graph) 與模型剪枝 (Pruning)。

## 2. 模型架構與實作說明 (Model Implementation)

### 2.1 使用模型：LeNet-5

本實驗在 PyTorch 與 TensorFlow 中實作了標準的 LeNet-5 架構：

輸入層：$32 \times 32 \times 3$ 影像。

卷積層 C1：6 個 $5 \times 5$ Filter，搭配 ReLU 激活。

池化層 S2：$2 \times 2$ Max Pooling。

卷積層 C3：16 個 $5 \times 5$ Filter，搭配 ReLU 激活。

池化層 S4：$2 \times 2$ Max Pooling。

全連接層 F5/F6：120 與 84 個神經元。

輸出層：7 個類別 (DeepPCB 瑕疵分類)。

### 2.2 框架實作細節

PyTorch 版本：使用 `nn.Module` 與 `nn.Sequential` 定義，並調用 CUDA 加速。

TensorFlow 版本：使用 Keras API 定義。實作了預設的「動態圖 (Eager Execution)」模式，以及透過 `@tf.function` 編譯的「靜態圖 (Static Graph)」模式。

## 3. 效能綜合比較 (Performance Comparison)

以下數據基於 RTX 4060 Laptop GPU 硬體環境測得，針對完整測試集進行壓測。

### 3.1 核心指標數據對照表

| 架構版本 (Framework) | 參數數量 (Params) | 延遲 (ms/img) | 吞吐量 (FPS) | 理論運算量 (FLOPs) | 準確率 (Test Acc) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Handcrafted (NumPy) | ~62,000 | 2.5237 | 396.25 | ~1.3M | ~93.2% |
| PyTorch (CUDA) | 61,751 | 0.1283 | 7791.73 | ~1.3M | ~94.0% |
| TF (Dynamic/Eager) | 61,751 | 0.1030 | 9710.64 | ~1.3M | ~93.8% |
| TF (Static Graph) | 61,751 | 0.0607 | 16477.20 | ~1.3M | ~93.8% |

### 3.2 結果分析與討論

1. 框架 vs. 手寫 (NumPy)：NumPy 版本雖然實作了 im2col 優化，但在 CPU 上的運算效率與 GPU 框架相比仍存在約 20 至 40 倍 的差距。這體現了 cuDNN 算子在並行計算上的絕對優勢。
2. FLOPs 估算：LeNet-5 在 $32 \times 32$ 輸入下，卷積與全連接層的總乘加運算數 (MACs) 約為 0.65M，換算為 FLOPs 約為 1.3M。

## 4. TensorFlow 靜態圖 vs. 動態圖分析

在 TensorFlow 的實驗中，我們重點對比了兩種執行模式：

* 動態圖 (Dynamic/Eager)：每一步運算即時回傳結果，開發偵錯容易，但 Python 調度開銷較大。
* 靜態圖 (Static/Graph)：透過 tf.function 將 Python 代碼轉換為編譯後的 Graph。

實驗發現：

* 推論時間：從動態圖的 0.1030 ms 降低至靜態圖的 0.0607 ms，效能提升約 1.7 倍。
* 吞吐量：靜態圖模式下的 FPS 達到了 16,477，證明在生產環境中進行圖編譯優化能顯著降低 Python Runtime 的延遲干擾。

## 5. 額外獎勵：模型壓縮 (Model Compression - Pruning)

我們應用了「權重剪枝 (Pruning)」技術對 TensorFlow 模型進行優化，將 50% 的參數設為零。

* 實作方法：使用 tensorflow_model_optimization 庫進行 Constant Sparsity 剪枝。
* 評估結果：
  * 儲存空間：模型存檔經 Gzip 壓縮後，體積減少約 40%。
  * 執行速度：在當前標準 GPU 算子下，剪枝後的推論延遲 (約 0.08ms) 略高於純靜態圖版本。這是因為 GPU 在進行稠密矩陣運算時，即便權重為零仍會參與計算，且額外的 Mask 遮罩運算引入了微小的開銷。
  * 結論：剪枝技術在減少存檔空間上非常有效，但推論加速則需依賴專用的稀疏矩陣加速硬體。

## 6. 訓練過程與預測結果 (Training & Evaluation)

### 6.1 訓練與驗證準確率曲線

根據 lenet_mlp_numpy.py 的訓練紀錄，模型在 30 個 Epoch 內展現了優異的收斂性：

* 收斂速度：由於使用了 Swish/ReLU 激活與 He 初始化，模型在第 10 個 Epoch 左右準確率即突破 90%。
* 過擬合控制：驗證集曲線與訓練集曲線貼合度高，顯示模型具有良好的泛化能力。

*(註：請參考 accuracy_comparison.png 圖表內容)*

### 6.2 預測結果總結

* 驗證集 (Validation Set)：Top-1 準確率約 94.2%。
* 測試集 (Testing Set)：最終預測準確率約 93.8%。

實驗證實，LeNet-5 架構在 DeepPCB 瑕疵分類任務上能達到極高的可靠性，且透過深度學習框架的編譯優化（特別是 TensorFlow 靜態圖），能在低延遲場景下實現極高效的即時檢測。

## 7. 結論 (Conclusion)

本實驗成功實作並對比了多個 LeNet-5 的版本。研究發現：

1. 框架優勢：TensorFlow 靜態圖在純推論吞吐量上表現最為優異。
2. 編譯開銷：動態圖雖然開發便利，但在高吞吐量需求下，靜態圖轉化是必要的步驟。
3. 手寫實踐：雖然 NumPy 版速度較慢，但對於底層 im2col 與 Swish 梯度的實作，有助於深入理解深度學習運算本質。