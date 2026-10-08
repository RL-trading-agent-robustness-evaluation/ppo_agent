# 代理實作 Prompt：PPO V11 因果動態訓練與三因子 reward

以下內容可直接交給實作代理。**本 Prompt 是未來工作交接，不代表產出此文件的工作已啟動程式修改、正式訓練或 known-period evaluation。**

---

你負責在既有 PPO 專案實作 V11。唯一新契約是 [V11 完整規格](../PPO_V11_ADAPTIVE_POLICY_REWARD_SPEC_ZH_HANT.md)，版本 `v11spec2_200k`。先讀完整規格，再讀 V10 規格、`configs/ppo_v10_nav_trend_reward.yaml`、V9 reward 契約及現有 wrappers/runners。規格為數值、時序、evaluation 與採用條件的 source of truth；本 Prompt 不授權更改其設計。

## 目標與交付

實作 chronological warm-start PPO、近期加權歷史窗口、VIX scheduler、混合 training reward、adaptive validation 與完整稽核。沿用既有 ledger/gate/五日規則。交付隔離的 V11 config、runner/模組、必要 tests、smoke 記錄、preflight、資源估計與使用說明。不得宣稱尚未執行的 formal matrix 已通過。

## 凍結值

```yaml
spec_version: v11spec2_200k
initial_training_start: '2010-01-01'
initial_training_end: '2021-12-31'
pre_2010_usage: causal_indicator_warmup_only
minimum_initialization_sessions: 252
episode_transitions: 63
episode_required_session_observations: 64
episode_boundary: time_limit_truncation_with_bootstrap
recency_half_life_sessions: 126
recency_exponential_mass: 0.80
recency_uniform_mass: 0.20
initial_requested_transitions: 200000
initial_expected_realized_transitions: 200704
fine_tune_transitions: 32768
initial_learning_rate: [0.0003, 0.00003]
fine_tune_learning_rate: 0.00003
keep_optimizer_state: true
reuse_old_policy_rollouts: false
scaler_fit: first_252_valid_post_2010_sessions_only
scaler_after_initialization: frozen
vix_reference_level: 20.0
vix_smoothing_sessions: 5
vix_availability_lag_sessions: 1
vix_period_exponent: 2
normal_period_sessions: 126
minimum_period_sessions: 21
maximum_period_sessions: 252
vix_forward_fill_limit_sessions: 3
period_rounding: floor_x_plus_half
weights: {nav: 0.30, relative: 0.20, trend: 0.50}
streak_beta: 0.10
streak_cap: 5
zero_breaks_streak: true
evaluation_reward: true_nav_log_return
evaluation_actions: deterministic
evaluation_learning: causal_online_fine_tuning_in_separate_clone
known_period: adaptive_known_period_confirmation_not_untouched_test
seeds: [0, 1, 2, 3, 4]
arms: [ppo_v11_vix_nav, ppo_v11_vix_mix, ppo_v11_fixed_nav]
formal_trajectories: 45
adoption_positive_fold_medians: 2
adoption_positive_pairs: 10
adoption_maximum_drawdown_deterioration_pp: 2.0
adoption_maximum_transaction_cost_increase: 0.10
```

保留 SB3 2.9.0、單一 env、CPU、MlpPolicy、n_steps 2048、batch 64、epochs 10、gamma .99、GAE .95、clip .2、advantage normalization、ent 0、vf .5、grad norm .5、[64,64] Tanh。初始學習率 schedule 只作用於初始化；後續 learning rate 必須固定，不因 resume 或 `learn()` 進度下降至其他值。

## 實作步驟

1. **Preflight 與 parent 核對。** 確認 V9 parent `f4988db5c98d9248533c4ca852cc48ca3f6b7aad`、實際 submodule、資料 schema 與 hashes；列出 V11 明示覆蓋的舊契約。確認 VIX 的來源與 availability；候選檔 `data/raw/VIXCLS_1990_2026.csv` 有 `observation_date,VIXCLS`。查核列覆蓋不等於允許在較早日期使用 future rows。
2. **Causal prefix 與 sampler。** 所有 market/features/events/cash/SPY/VIX 讀取均受 cutoff 限制。2010 前只暖機 indicators。scaler 在初始 252 sessions 擬合後固定。列舉 63-transition 窗口，使用 `p_i=.8*2**(-a_i/126)/sum_j(2**(-a_j/126))+.2/N` 有放回抽樣；age 依 SPY session index 與窗口最後 valuation 日計算。短窗口為 time-limit truncation；保留 terminal observation 與 bootstrap。
3. **Reward wrapper。** 直接由 unshaped NAV 與既有 SPY reference 計算 `r_relative=r_nav-r_spy`；`b=.1*max(min(streak,5)-1,0)*max(r_nav,0)`；`r_trend=r_nav+b`；`r_train=.3*r_nav+.2*r_relative+.5*r_trend`。zero/negative reset streak，episode reset 亦 reset。NAV arms 只交付 NAV reward。禁止 relative wrapper 重複扣 benchmark。
4. **Scheduler。** 對 SPY calendar 正向填補 VIX 最多三 sessions，非正/非有限視無效，第四次缺失停止。使用 `t-5..t-1` 五個 aligned values；`P=clip(floor(126*(20/mean_vix)**2+.5),21,252)`。每日重新計算，elapsed 達 P 更新，最多每日一次。fixed-nav 僅改 P=126。
5. **Chronological controller。** 各 fold/seed/arm 獨立從 2010 推進；252-session 收盤初始化，此前 gate-only。close t 先以舊模型決策，再以 cutoff=t 的 training clone 更新；新模型首次用於 close t+1 決策，最早 open t+2 成交。最後計分日不做無後續影響更新。保留 optimizer、scaler、RNG 與 update clock，fresh rollout，不 replay 舊 advantages。保存完整 checkpoint resume state。
6. **Adaptive evaluation。** train/validation boundary 保留模型與 scheduler，按既有 scoring 契約建立計分帳戶；不在中途更新 reset NAV、持倉或五日規則。evaluation actions deterministic，delivered reward 僅 NAV；允許 isolation clone 用已觀測 validation prefix 訓練。folds validation 為 2015–2016、2017–2018、2019–2021；其 train prefixes 分別結束 2014、2016、2018，全部從 2010 起。
7. **對照與報告。** reward：vix-mix/vix-nav；schedule：vix-nav/fixed-nav，共用已完成 vix-nav 結果。45 trajectories，各初始 200,704 加每次 32,768；不是總共每 cell 200k。每個對照報全部 15 deltas 與四門檻。MDD/cost 先取 fold 內五-seed median，再取三-fold median。control cost=0 時 treatment 必须=0。缺 paired cell 標 INCOMPLETE。兩對照皆過才採用完整方案；不自動發明 fixed-mix。
8. **驗證與交付。** 執行完整規格第 8 節 tests。通過後執行 fold_1/seed0 三-arm smoke，初始化 20,000 requested/20,480 expected，再一次 32,768 更新；強制 smoke clock 與 formal 隔離。交付 formal 資源預估與命令，但不要因讀到本 Prompt 就自動啟動 45 trajectories 或 2022–2025。

## 必要驗證與紀錄

- 日期邊界、64-session/63-transition 窗口、252-session 初始化與 future sentinel。
- sampler 機率總和誤差 ≤1e-12、半衰期 component 比值、RNG 重現。
- VIX 表：15→224、20→126、25→81、30→56、40→32、50→21；clamp、rounding、lag、missing、daily rescheduling。
- streak/reset/zero/cap、混合公式、same-action ledger identity；完整 episode reward identity 誤差 ≤1e-10。
- activation timing、scoring/train 隔離、optimizer continuation、固定 scaler/LR、fresh rollout、checkpoint resume、自然終止與 time-limit bootstrap。
- paired aggregation、門檻等號、zero-cost、失敗記錄及 append-only behavior。

保存每個 step 的 NAV/SPY/relative/trend/bonus/training reward 與 action/gate/execution 診斷。每次更新保存 cutoff、trigger/activation/first-fill dates、VIX 原值/填補值、elapsed/P、window probabilities/sample ages、model/optimizer/scaler/RNG hashes、transitions、LR、耗時及失敗。所有 baseline/績效、gate diagnostics、compute 與 component contribution 依完整規格輸出。

所有 artifacts 使用 `experiments/v11_ppo/`、`artifacts/ppo_v11/`、`reports/v11_ppo/`、`tensorboard/ppo_v11/`。不覆寫 V9/V10，不把已開啟 2022–2025 稱作 untouched test。敏感度只在 development，另標 exploratory，不沿用 formal v11spec2_200k 名稱改參數。

## 禁止事項與停止條件

禁止 future data、backfill、全樣本 scaler、evaluation shaping、修改 ledger/gate/交易成本/動作/observation 維度、挑 seed、混用舊實驗、把 shaping 當收益、靜默丟棄失敗、把排程對照稱 compute-matched。

如遇來源或可得時點無法確認、必要資料缺失、VIX 超過三 sessions 缺值、parent 契約未明示衝突、空窗口、非有限模型/loss、identity/隔離測試失敗、無法完整 resume 或 config 與凍結值不符，停止受影響 run，保存 append-only failure 與可定位原因；不得自行改規格或用不同資料補齊正式比較。

## 最終回覆格式

依序提供：實作內容與檔案連結、tests 與 smoke 證據、未執行項目、資源估計/可用命令、尚存限制。明確說明 50% trend coefficient 不等於 50% bonus contribution；參數是工程預設，研究來源支持原則，不證明最優數值。若僅完成部分，精確標記其狀態，不宣稱 formal adoption 通過。


**2026-10-07 使用者修訂：** 本版本初始化 requested transitions 改為 200,000，SB3 實際 200,704；每次 fine-tuning 仍為 32,768。原 V10 目錄中的 v11spec1 文件保留為歷史版本。
