# PPO V11 — 200k chronological adaptive policy

本目錄實作 [V11 規格](docs/PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md) 與 [代理 Prompt](docs/prompts/PPO_V11_IMPLEMENTATION_PROMPT_ZH_HANT.md)。使用者修訂版本為 `v11spec2_200k`：初始化請求 **200,000** transitions，SB3 完整 rollout 為 **200,704**；每次 fine-tune **32,768**，不是整條多年 trajectory 的總步數。

保留 V10 的 ledger、SPY features、SMA-200 gate、五日規則與 PPO hyperparameters。新增程式位於 `src/victim_ppo/v11/`；沿用檔案與來源 commits/hashes 見 `SOURCE_MANIFEST.json`。未複製 Git metadata、虛擬環境或舊 checkpoints/results；local licensed processed data 仍由 `.gitignore` 排除。

## 執行

已安裝依賴的 Python（SB3 2.9.0）即可運行。此工作區可直接重用鄰近 V10 runtime；不需要重複安裝或複製其 `.venv`。runner 自行加入本地 V11 與 victimagent source，確保載入新程式。

```powershell
# 在 ppo_v11 目錄執行
$ppoPython = '..\ppo_v10\.venv\Scripts\python.exe'
& $ppoPython scripts/run_ppo_v11.py --preflight
& $ppoPython -m pytest tests
& $ppoPython scripts/run_ppo_v11.py --smoke --arm ppo_v11_vix_nav --tag smoke1
& $ppoPython scripts/run_ppo_v11.py --smoke --arm ppo_v11_vix_mix --tag smoke1
& $ppoPython scripts/run_ppo_v11.py --smoke --arm ppo_v11_fixed_nav --tag smoke1
# 正式 45 trajectories（可能有大量更新；只在準備執行全矩陣時呼叫）
.\scripts\run_ppo_v11_matrix.ps1 -Python $ppoPython -Tag v11_200k_1
```

單一 trajectory：`--run --arm ARM --fold fold_1 --seed 0 --tag UNIQUE_TAG`。所有輸出以 tag 與 namespace 隔離，已存在的結果拒絕覆寫。`--aggregate --tag TAG` 保存兩對照摘要；缺任何 paired seed 為 `INCOMPLETE`。

VIX period `clip(round_half_up(126*(20/mean_VIX)^2),21,252)`，平均僅用前五 sessions，缺值 forward-fill 最多三 sessions。63-transition episodes 按 endpoint age 的 80% exponential + 20% uniform 分布抽樣。初始化 scaler 後永不 refit。訓練與計分帳戶分離；close t 先決策後更新，新模型從 close t+1 決策／open t+2 成交開始生效。

## Resume 與 known period

每個完成更新保存 model/optimizer、clock、scaler 指紋、Python/NumPy/Torch RNG；validation 更新後亦保存完整 scoring environment、ledger、五日狀態與 observation。僅可使用同 run 目錄內可信的 `resume_PHASE_INDEX`：

```powershell
& $ppoPython scripts/run_ppo_v11.py --run --arm ppo_v11_vix_nav --fold fold_1 --seed 0 --tag TAG --resume artifacts/ppo_v11/formal/RUN_LABEL/resume_validation_INDEX
```

Resume files 是可信本地 pickle，不能載入第三方檔案。只支持完整 checkpoint 邊界恢復；如已存在未完成的 update/output，安全停止，保留失敗並以新 tag 從頭重跑，禁止刪紀錄後混入正式結果。

2022–2025 是已開啟的 adaptive known-period confirmation。CLI 必須使用明確 `--known-period`，且先有同 tag 完整 45-cell development summary；不會由 matrix 自動執行。此期間只循時學習已觀測 prefix，不供調參或挑 seed。

### 使用者直接授權的 2022–2025 評估（2026-10-07）

使用者另行要求在 2022–2025 評估混合 reward 模型，允許測試與後續 deployment 持續動態更新，並指定全部五 seeds。此為獨立的 `direct adaptive known-period diagnostic`，不聲稱已完成 development selection；明確 CLI `--direct-known-period` 不要求 45-cell selection matrix，但仍要求三-arm smoke 與全部 causal/data preflight。

```powershell
& $ppoPython scripts/run_adaptive_2022_2025.py --workers 2 --tag adaptive_20261007_1
```

每個 seed 先循時在 2010–2021 prefix 完成 200k 初始化及排程更新，保存 `pre_2022_model.zip` 和 freeze provenance，再以新計分帳戶執行 2022–2025 adaptive evaluation。保留全部五 seeds，禁止使用此期間調整參數。最後保存 `final_model.zip`、完整 `resume_deployment_INDEX` 帳戶/optimizer/clock/RNG 狀態及 `deployment_contract.json`。後續部署可沿用同一政策更新，但新增資料必須從保存的 close 接續，不能將 data-boundary termination 解讀為帳戶 reset。

## 驗證與限制

正式初始化為 200k，smoke 初始化為 20k 並強制一次完整 32,768-step 更新，接著短期 NAV-only scoring。smoke 不列入正式結果。`reports/V11_IMPLEMENTATION_CHECKS.md` 記錄實際驗證狀態；formal 與 known period 未執行時不宣稱已完成。

初始訓練 prefix 在第一個 252-session cutoff 初始化，再循日更新；初始化以前沒有學習或計分，策略定義為既有 gate-only baseline。scoring 從各 fold validation 起點的新帳戶開始；train-prefix 只推進學習時鐘，無須將暖機交易帳戶帶入 validation。

計算成本隨更新次數增加。每條正式 trajectory 總 transitions = `200,704 + 32,768 × 更新數`；VIX 對照與固定排程不保證 compute matched。50% trend component 是 NAV-plus-streak 的係數，加權 bonus 上限僅為正 NAV 的 20%。

## GitHub 與已完成結果

V11 發布於同一 `ppo_agent` repository 的獨立 `v11-adaptive-policy` 分支；保留原版本歷史。GitHub clone 請使用 `git clone --recurse-submodules`，確保取得既有 `victimagent` commit；本機既有copied source內容與該commit一致。

五 seeds 0–4 的 adaptive 2022–2025 評估已完成；完整報告 [adaptive_20261007_1](reports/v11_ppo/adaptive_2022_2025_adaptive_20261007_1.md)。中位數 wealth1.4509065、CAGR9.7724%、MDD-19.0903%；2/5勝原B&H。這是 adaptive known-period diagnostic，45-cell development矩陣尚未執行。

資料、模型checkpoints及runtime artifacts依既有ignore rules不入Git；GitHub包含source、規格／Prompt、結果摘要、驗證與provenance。licensed datasets需要在本機另行準備，不能僅clone就直接重跑。
