# PPO V12：定額 B&H 基準比較

本次只計算 2022–2025 每月定額 B&H，與既有 V11 五 seeds 比較，不訓練、不重新推論。依賴相鄰 `ppo_v11` 的唯讀資料、ledger 程式與既有結果；不複製模型或 Python 環境。

- [研究與實作規格](docs/PPO_V12_MONTHLY_BH_SPEC_ZH_HANT.md)
- [代理 Prompt](docs/prompts/PPO_V12_IMPLEMENTATION_PROMPT_ZH_HANT.md)
- [比較報告](reports/v12_benchmark/comparison_2022_2025.md)
- [驗證](reports/validation/VALIDATION.md)
- [Devlog](dev/devlog.md)

從 `C:\03_College\Grad_proj` 執行：

```powershell
& '.\ppo_v10\.venv\Scripts\python.exe' -B -m unittest discover -s ppo_v12/tests -v
& '.\ppo_v10\.venv\Scripts\python.exe' -B ppo_v12/scripts/compare_existing.py
```

比較入口拒絕覆寫已完成報告；讀既有報告即可檢查結果。失敗保存 attempt traceback 與產物。若需再次計算，先建立獨立輸出版本，不刪除既有紀錄。

48 次含成本支出合計 $1m；fractional shares；剩餘現金生息、股息留現金。原 B&H 與模型仍為整股。**adaptive known-period diagnostic**，不是 untouched test，也不是模型改善證據。未來 200k／四種子訓練僅登記於 Prompt，等待後續指示。

## V12 完整訓練已授權

使用者於 2026-10-08 明確授權完整訓練；本階段以 [執行附錄與接手 Prompt](docs/PPO_V12_FULL_TRAINING_ADDENDUM_ZH_HANT.md) 取代先前尚未授權訓練的停止條件。五 seeds 0–4，500k requested / 501,760 actual 初始化，後續每次 32,768；原 V11 reward 與更新政策不變。隔離 engine 位於 training，背景 supervisor scripts/run_full_training.py 保存凍結 protocol、三-arm smoke 與完整比較。

執行狀態：reports/full_training/full500k_20261008_1/status.json。最終報告完成後位於同目錄 COMPARISON.md／JSON；在完成前不宣稱績效或挑選模型。原 SOURCE_MANIFEST 與比較報告保留為前階段交付快照，新訓練來源以此次 protocol／ENGINE_CHANGES 為準。
