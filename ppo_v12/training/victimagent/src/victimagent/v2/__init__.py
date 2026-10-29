"""DQN V2 controlled-optimization tooling.

Everything here selects models using train and rolling validation only. The
2022-2025 known-period window is never opened for selection; see
``docs/v2_optimization_contract_v1.md`` and ``configs/v2_rolling_folds.yaml``.
"""
