# v1.0.0 驗證範圍

2026-11-09，本機 Windows、64-bit Python 3.12.10、PyTorch 2.14.1+cpu、NumPy 2.5.2。

## 未訓練權重證據

三份隨附模型由 `tools/init_models.py` 單純初始化後儲存。該程式沒有 optimizer、訓練迴圈或載入既有權重。

逐一比對每個 tensor 與相同 seed 42 的新模型，在本機完全相同。三份 checkpoint 與 manifest 均為 `status=untrained`、`episodes=0`、`optimizer_updates=0`；雜湊見 `models/manifest.json`。

| 模型 | 可訓練參數 |
| --- | ---: |
| GRU | 27,205 |
| LSTM | 35,525 |
| Transformer | 71,237 |

## 已通過的本機檢查

`python -m unittest tests.test_baselines -v`：10 項通過。

- 實際使用三種不同 PyTorch backbone，重設記憶可重現初始輸出。
- 模型動作進入連續環境，下一次觀察含上次動作；預覽與評估不改變模型權重。
- Transformer 的歷史最多 32 步，修改未來 token 不影響較早 token 的表示。
- LSTM 的 hidden、cell 皆在 rollout 邊界 detach。
- checkpoint round-trip、跨架構拒絕、拒絕覆寫。
- 每個架構在暫存目錄執行 2 回合、6 次 optimizer 更新，確認 backbone 確實改變、可存讀並評估；測後三份隨附模型 SHA256 不變。
- 斜向位移、碰撞、不可達地圖拒絕。
- 隨機起點排除目標格，100 次 reset 均未出生於目標範圍。
- 三個 CLI 皆可從不相關工作目錄啟動並完成 check。

`python -m unittest tests.test_viewer -v`：1 項通過，涵蓋三個架構的 Tk 視窗建立、繪製、模型控制一步、重設。

## 解讀限制

這些結果證明程式、模型介面及訓練機制可以運作，不代表未訓練模型會解迷宮，也不構成三種架構的學習成效排名。

Windows 本機已執行上述測試。GitHub Actions 提供 Windows / Apple Silicon macOS 的 headless 測試；實際 CI 結果請看 repo 的 Actions。macOS 桌面 GUI 尚未在實體 Mac 人工驗證。
