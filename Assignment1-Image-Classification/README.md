# 環境需求

* Python 3.8+
* PyTorch (支援 CUDA 加速)
* OpenCV, Numpy, Matplotlib, Scikit-learn

# 如何執行

1. 資料準備： 確保資料集放置於 `../DeepPCB`，結構包含 `train/`, `valid/`, `test/` 資料夾。

2. 執行訓練：
```
python image_classification_gpu.py
```

1. 評估流程：

    * 程式會自動讀取影像並根據標籤進行局部裁剪。
    * 依序訓練 Perceptron、MLP。
    * 自動儲存學習曲線圖檔 (`.png`)。
    * 最後運行 Random Forest 並在終端機輸出三者的 Top-1/Top-5 對比表格。

# 標籤校驗
程式啟動後會輸出 `Label Mapping Check`，顯示偵測到的類別數量與樣本分佈，用於確保數據載入過程無誤。
