# 未訓練模型

三份 `.pt` 均由 `tools/init_models.py` 使用 seed 42 隨機初始化，未執行 optimizer：`episodes=0`、`optimizer_updates=0`。

`manifest.json` 列出各自架構、參數量與 SHA256，預設載入會驗證雜湊。單元測試會重新初始化相同 seed，逐一比對每個 tensor；容許 Windows／macOS 數值函式庫造成的末位差異（atol=1e-6、rtol=1e-5）。跨 PyTorch 版本初始化實作可能不同；重建檢查使用 manifest 記錄的版本。

使用者訓練的權重請放 `outputs/` 並以 `--checkpoint` 明確指定。預設仍載入這裡的未訓練權重。
