# Fly Maze Baselines：GRU / LSTM / Transformer

三個**未訓練**的連續迷宮導航版本，使用相同環境、感測器、動作與 Actor-Critic 訓練流程，供後續實驗比較。這是獨立專案，基於 [Fly Maze](https://github.com/Sloth0528/fly-maze) 的連續環境整理。

版本 `1.0.0`。隨附的三份權重都是 seed 42 的隨機初始化，**訓練回合 0、optimizer 更新 0**。預覽及評估不會更新權重。未訓練模型可能直走、撞牆或繞圈，不能當成已學會走迷宮。

| 版本 | 執行檔 | 記憶結構 | 預設權重 |
| --- | --- | --- | --- |
| GRU | `run_gru.py` | 64 維 hidden state | `models/gru_untrained.pt` |
| LSTM | `run_lstm.py` | 64 維 hidden + cell state | `models/lstm_untrained.pt` |
| Transformer | `run_transformer.py` | 最近 32 步，2 層、4 heads、64 維，因果注意力 | `models/transformer_untrained.pt` |

三者均為一般神經網路模型，沒有接入果蠅連接組。參數量不同；相同 hidden size 不代表相同計算量或容量。架構、參數量、初始化 seed 與 SHA256 在 [models/manifest.json](models/manifest.json)。

## Windows 安裝與執行

使用 **64 位元 Python 3.12**。下載 repo 後，在專案資料夾開啟 PowerShell：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe run_gru.py
```

預覽視窗按 **Play / Pause** 開始。替換執行檔即可開啟另外兩版：

```powershell
& .\.venv\Scripts\python.exe run_lstm.py
& .\.venv\Scripts\python.exe run_transformer.py
```

安裝完成後，也可雙擊 `START_GRU.cmd`、`START_LSTM.cmd`、`START_TRANSFORMER.cmd`。

## macOS 安裝與執行

此依賴設定支援 **Apple Silicon（M1/M2/M3/M4 等）的原生 arm64 Python 3.12**。Intel Mac 與 Rosetta x86_64 不在此版本的支援範圍。

建議安裝 [python.org 的 Python 3.12](https://www.python.org/downloads/macos/)，再於專案資料夾執行：

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m tkinter
./.venv/bin/python run_gru.py
./.venv/bin/python run_lstm.py
./.venv/bin/python run_transformer.py
```

`python -m tkinter` 應開啟小視窗，可關閉後繼續。若缺少 Tkinter，需使用含 Tcl/Tk 的 Python；pip 不能安裝系統 Tkinter。

也可 `chmod +x START_*.command` 後雙擊對應的 `.command`；或 `bash START_GRU.command`。Mac 安裝指令使用 PyPI。

## 確認權重與不開視窗評估

以下用 `python` 表示上述 `.venv` 的 Python 執行檔。

```bash
python run_gru.py check
python run_lstm.py check
python run_transformer.py check
python run_gru.py evaluate --episodes 20 --max-steps 160 --random-start --seed 40007 --output outputs/gru_before.json
python run_lstm.py evaluate --episodes 20 --max-steps 160 --random-start --seed 40007 --output outputs/lstm_before.json
python run_transformer.py evaluate --episodes 20 --max-steps 160 --random-start --seed 40007 --output outputs/transformer_before.json
```

`check` 驗證預設檔案 SHA256、載入模型並實際執行一步。視窗與終端會顯示 UNTRAINED、episodes=0、updates=0。評估預設採 deterministic/greedy 動作；加 `--sample` 才會取樣。

## 從零訓練

只有明確執行 `train` 才會訓練。輸出到另外的檔案，不覆寫隨附的未訓練模型：

```bash
python run_gru.py train --episodes 1000 --max-steps 160 --random-start --seed 42 --output outputs/gru_seed42.pt
python run_lstm.py train --episodes 1000 --max-steps 160 --random-start --seed 42 --output outputs/lstm_seed42.pt
python run_transformer.py train --episodes 1000 --max-steps 160 --random-start --seed 42 --output outputs/transformer_seed42.pt
```

這是可以執行的訓練起點，1000 回合不保證學會導航。訓練會同時產生 `.csv`，記錄 reward、成功、碰撞與 loss。請使用多個 seed、獨立測試地圖與相同評估預算來比較成效。

載入自己訓練的模型：

```bash
python run_gru.py preview --checkpoint outputs/gru_seed42.pt
python run_gru.py evaluate --checkpoint outputs/gru_seed42.pt --episodes 100 --random-start --seed 50007 --output outputs/gru_after.json
```

LSTM、Transformer 替換執行檔與 checkpoint 名稱即可。每個架構的執行檔會拒絕其他架構權重。`train` 每次從指定 seed 的隨機初始化開始，不提供續訓 optimizer 的功能。

若只要另外產生未訓練權重：

```bash
python run_lstm.py init --seed 123 --output outputs/lstm_untrained_seed123.pt
```

## 地圖與環境

共同輸入 31 維：13 條射線距離、13 個可見食物指標、朝向 cos/sin、上次兩個動作、碰撞狀態。輸出 `[throttle 0..1, steering -1..1]`，控制連續座標與旋轉，支援斜向與弧線移動。模型不接收整張地圖或預先計算的路徑。

預設使用原本的小地圖。可用 `--difficulty easy` / `medium` / `hard` 產生地圖；訓練時加 `--vary-maze` 每回合換圖。`--maze examples/open_maze.txt` 可載入自訂 TXT，接受 `#` 牆、`.` 路、`S` 起點、`G` 目標；需要封閉邊界且起點可達目標。`--maze` 與 `--difficulty` 擇一。

```bash
python run_transformer.py preview --maze examples/open_maze.txt --sample
python run_gru.py train --difficulty easy --vary-maze --random-start --episodes 1000 --seed 42 --output outputs/gru_easy.pt
```

指定地圖只會更換測試環境，不會讓舊權重自動學會新地圖。訓練預設 Adam lr=0.0003、gamma=0.99、rollout=32、梯度裁切 0.5、CPU 單執行緒；GRU/LSTM 用截斷 BPTT，Transformer 每步重新編碼最近 32 個原始觀察值。

## 程式分工與測試

| 檔案 | 用途 |
| --- | --- |
| `run_gru.py` / `run_lstm.py` / `run_transformer.py` | 三個獨立入口 |
| `maze_baselines/models.py` | 三種模型、記憶、動作、存讀權重 |
| `maze_baselines/environment.py` | 連續移動、碰撞、感測、獎勵 |
| `maze_baselines/experiment.py` | 共用訓練與評估 |
| `maze_baselines/viewer.py` | 可視化、實際權重狀態、控制輸出 |
| `tools/init_models.py` | 僅產生隨機權重，拒絕覆寫現有預設檔 |
| `tests/` | 初始化證據、注意力遮罩、訓練更新、碰撞、CLI、GUI smoke |

```bash
python -m unittest tests.test_baselines -v
python -m unittest tests.test_viewer -v
```

GUI 測試需要桌面。測試中的少量訓練只使用暫存目錄，測後檢查隨附權重雜湊不變。見 [VALIDATION.md](VALIDATION.md) 的實測範圍。

實作參考 PyTorch 官方文件：[GRUCell](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRUCell.html)、[LSTMCell](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTMCell.html)、[TransformerEncoder](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)。
