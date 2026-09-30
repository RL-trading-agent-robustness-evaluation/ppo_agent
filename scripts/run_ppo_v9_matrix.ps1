param([int]$Threads = 1)

$ErrorActionPreference = "Stop"
$arms = @("ppo_v9_gate_nav", "ppo_v9_gate_relative")
$folds = @("fold_1", "fold_2", "fold_3")
$seeds = 0..4

foreach ($arm in $arms) {
    foreach ($fold in $folds) {
        foreach ($seed in $seeds) {
            & .\.venv\Scripts\python.exe scripts\run_ppo_v9.py --run `
                --arm $arm --fold $fold --seed $seed --threads $Threads
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}
& .\.venv\Scripts\python.exe scripts\run_ppo_v9.py --aggregate
exit $LASTEXITCODE
