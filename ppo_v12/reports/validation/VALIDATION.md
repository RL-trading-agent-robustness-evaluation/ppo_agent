# V12 驗證報告

2026-10-08；結果 **PASS**。本次新增 PPO 訓練 transitions=0，模型推論次數=0。

- 最終 9 個單元測試全部通過，涵蓋 48 月排程與尾差、常數／變價價格路徑、成本與 NAV 守恆、lagged interest、股息 entitlement／結清、fractional split、未來價格 sentinel、缺月／資金不足停止，以及基準不使用 policy observation。
- 五個 V11 seeds 0–4 都有原始 PASS／technical_pass；逐一核對資料、validation ledger 與 final checkpoint hashes，重算 wealth／CAGR／MDD／成本，驗證 NAV telescoping 及 log-reward identity。
- 定額及原 B&H 與五模型完全使用相同 1,002 transitions，初始收盤 2022-01-03、終點收盤 2025-12-31；原整股 B&H 在五個 records 中的指標皆重現。
- 48 筆定額購買，首筆 2022-01-04；本金支出與 $1m 的差異僅為小於 1e-8 USD 的浮點數誤差。成本 $999.000999；現金非負，每 session NAV 組成與交易現金守恆通過。
- V12 使用獨立 fractional ledger，原 V11 account features 只支援整股，因此月買基準不呼叫模型 observation composer；保留原現金／事件時序。
- 執行結果與來源 hashes 見 [完整 JSON](../v12_benchmark/comparison_2022_2025.json)。完成後由 `verify_delivery.py` 再核對產物 hashes、日曆、預算與最終資料，生成驗證 JSON 與交付 manifest。

失敗紀錄全部保留：首次合成 fixture 缺欄位（[說明](initial_fixture_failure.md)）；整合 attempt 1 的 VIX config 缺 hash 欄位（改以既有 record 凍結值核對）；attempt 2 的整股 account feature 限制（隔離基準 observation）；attempt 3 完成。原 V11 未修改，沒有透過改參數略過失敗。

Logs：[8 tests 初次通過](unit_tests_attempt1.log)、[9 tests 最終通過](unit_tests_attempt2.log)、[失敗 1](comparison_attempt1.log)、[失敗 2](comparison_attempt2.log)、[成功](comparison_attempt3.log)。執行 attempts 原始 traceback 保存在 `reports/v12_benchmark/attempt_*.json`。

未執行：新模型 checkpoint reload、訓練 smoke、正式三fold／五seed reward 與排程矩陣、部署。這些不是本次基準比較的必要工作，正式採用 **NOT_EVALUATED**。
