# 原始碼來源

連續環境 `maze_baselines/environment.py`、幾何 `geometry.py` 與地圖產生器 `maze_layouts.py` 整理自使用者自己的 [Sloth0528/fly-maze](https://github.com/Sloth0528/fly-maze) 本機工作區，在 2026-11-09 建立此獨立比較專案。

新的 GRU / LSTM / Transformer 權重均從頭初始化，沒有載入舊專案的已訓練權重、果蠅連接組資料或外部預訓練模型。

此專案額外排除目標格作為隨機起點，避免開始時已位於目標範圍而高估成功率。
