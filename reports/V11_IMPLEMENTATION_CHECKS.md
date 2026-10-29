# V11 implementation verification — 2026-10-07

**版本：v11spec2_200k；正式訓練與 known period 尚未執行。**

## 結果

- 19 tests passed；完整輸出：`pytest_stdout.log`。
- Preflight PASS：市場及股利 hashes 與 V10 相符，VIX 與 copied-source manifest 亦已核對。
- 三-arm smoke 均完成 20,480 初始化 + 32,768 fine-tune = 53,248 transitions。
- 實際 model/optimizer checkpoint reload、finite losses、固定 scaler、sampling、自然計分終止與 NAV identity 均通過。

| Arm | Smoke seconds | NAV identity error | Training response checks | Short scoring response |
|---|---:|---:|---|---|
| ppo_v11_fixed_nav | 291.7 | 2.01e-16 | [True, True] | False |
| ppo_v11_vix_mix | 292.3 | 2.01e-16 | [True, True] | False |
| ppo_v11_vix_nav | 224.8 | 2.01e-16 | [True, True] | False |

## 限制與尚未執行項目

- Smoke 為短期 scoring，未必在該 20-transition 窗口呈現 deterministic action responsiveness；訓練樣本的 response 診斷另列，禁止將低 action variation 當作正式績效證明。
- Smoke 在實作過程中啟動，其 records 保存當時 provenance。之後收尾的 preflight 計算量估計、formal smoke gate、logging 與 action-lock helper 由最終 tests 覆蓋；不可把三份 smoke hashes 當作最終完整 source tree hash。最終 source hashes 另存 JSON。
- Resume 已驗證可信本地完整 checkpoint 的 hash、model/optimizer 與 ledger/五日狀態 roundtrip；不支援中斷在 update 內的半完成 optimizer 恢復。
- 未執行 200k formal cell、45-cell matrix、development 敏感度或 2022–2025 adaptive confirmation，尚無 V11 採用結論。

## 正式預算

依目前 development sessions/VIX，每條 trajectory 的 fine-tunes：fold_1 VIX/fixed = 11/11；fold_2 = 15/15；fold_3 = 24/21。

45 trajectories 總預期 **33,116,160 transitions**，包含 45 次 200,704 初始化；200k 不含後續更新。Smoke 的實際速度不可直接當作多年正式資料的 ETA，因 reset 與歷史窗口長度會改變成本。

執行與 resume 命令見 [README](../README.md)。機器可讀驗證證據見 [JSON](V11_IMPLEMENTATION_CHECKS.json)。
