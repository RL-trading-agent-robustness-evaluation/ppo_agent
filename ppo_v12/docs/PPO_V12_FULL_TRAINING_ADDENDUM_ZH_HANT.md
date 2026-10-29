# V12 完整模型訓練與版本比較：執行附錄

2026-10-08；使用者已明確授權完整訓練，允許增加 seeds／timesteps。本附錄取代原 V12 規格與 Prompt 的「尚未授權訓練」停止條件；既有基準比較與原始文件仍保留，不覆寫歷史。

## 凍結方案

重新訓練五個混合 reward 模型，seeds 0–4，初始化請求 **500,000** transitions，2048-step rollout 實際 **501,760**。每次因果更新 **32,768**；目前日曆預計 24 次 pre-2022 與 9 次 2022–2025 更新，每 seed 全程 **1,583,104** transitions，五 seeds 合計 **7,915,520**（另加必要 smoke）。增加初始化預算是預先登記的計算預算對照，不是已證實改善，不依已知期間結果繼續調參或延長訓練。

保留 V11 全部政策、reward、事件及成交語義：2010 起、初始 2021 截止、252-session 初始化及固定 scaler、63-transition episodes、126 半衰期 80/20 抽樣、VIX 五值均值及 21–252 更新時鐘、fresh on-policy rollouts、warm-start optimizer、30/20/50 reward、beta=.1、cap=5、既有 gate／五日限制／整股／10bps。初始化學習率仍依該次实际步數線性下降，fine-tune 固定 3e-5。500k 相對 200k 也改變初始化 schedule 的持續時間，須一併揭露。

使用 `ppo_v12/training` 獨立 engine，唯有版本名稱、輸出 namespace 與初始化預算改變。原 parent manifest 的共同環境及資料逐檔 hash 核對；保留 ENGINE_CHANGES、config、preflight、三 arms smoke、所有失敗。開始五 seeds 前必須通過測試與 smoke；訓練不接續舊模型，而從每 seed 的歷史 prefix 重新初始化。

## 評估與比較

固定 2022-01-03 收盤至 2025-12-31 收盤、1,002 transitions，deterministic 決策並允許已觀測歷史因果更新；不能重設 scoring ledger、持有天數、optimizer 或時鐘。保留每 seed pre-2022、各次更新、final 及 deployment checkpoints。不得自動部署或挑最佳模型；交付全部模型及中位數報告。

同時比較：V12 月買 B&H、原整股 B&H、V11 200k 五 seeds、既有 V4／V9／V10 known-period 結果。新模型與 V11 按同 seed 提供 paired log-wealth 差、CAGR／MDD／成本差、正差數、中位數與範圍；對兩基準列勝數。舊版本比較須列各自資料、訓練 cutoff、reward、budget、static/adaptive 差異，V4 的不同市場資料 hash 不視為 matched control。

標為 **adaptive known-period training-budget comparison**。2022–2025 已開啟，不能叫 untouched test，不使用該期間挑參數／seed／checkpoint。此執行為完整五模型的直接比較，不是正式三fold reward／排程選擇矩陣；原正式採用門檻仍保留、狀態 NOT_EVALUATED。既有模型版本跨方案比較僅描述，不據此宣稱 reward 或排程的單獨因果改善。

## 執行與監控

先執行隔離測試與 preflight；背景 supervisor 依序完成三 arms smoke，再以兩 workers 執行五 seeds。開始前檢查沒有同 tag 活動工作。保存 PID、每 seed log、逐次 cutoff／原因／VIX／模型 hashes／sampling／actual transitions、attempts、狀態與 heartbeat。

健康進度不反覆通知；全部完成且 technical_pass／有限 NAV／hashes／一致日曆／基準重現檢查通過後，只通知一次並停用完成 monitor。失敗保留 logs；停止派發新的 seeds，讓其他健康工作完成，不任意終止。需要使用者介入才通知。

## 接手代理 Prompt

你已獲使用者完整訓練授權。讀本附錄、原 V12 規格及訓練 protocol；凍結 seeds0–4、500k 初始化、32,768 fine-tunes 和所有 V11 經濟規則。使用隔離 engine，先通過單元測試及三 arms smoke，啟動單一 supervisor；不得重複派發、覆寫產物、根據 2022–2025 調參或提前挑選結果。完成後驗證所有五 records、ledger/checkpoint hashes、實際步數及既有基準，輸出全版本 Markdown/JSON，更新 devlog，保留全部模型與部署狀態，停止。不得自動交易、部署或將直接 known-period 比較稱正式採用通過。
