param([string]$Python = 'python', [string]$Tag = 'v11_200k_1')
$ErrorActionPreference = 'Stop'
$runnerPath = Join-Path $PSScriptRoot 'run_ppo_v11.py'
& $Python $runnerPath --preflight
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
foreach ($foldId in @('fold_1','fold_2','fold_3')) {
    foreach ($seedId in 0..4) {
        foreach ($armId in @('ppo_v11_vix_nav','ppo_v11_vix_mix','ppo_v11_fixed_nav')) {
            & $Python $runnerPath --run --arm $armId --fold $foldId --seed $seedId --tag $Tag
            if ($LASTEXITCODE -ne 0) { throw "V11 cell failed: $armId $foldId $seedId" }
        }
    }
}
& $Python $runnerPath --aggregate --tag $Tag
if ($LASTEXITCODE -ne 0) { throw 'Aggregation failed' }
