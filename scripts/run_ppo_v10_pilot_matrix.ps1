param([int]$Threads = 1, [int]$MaxParallel = 2)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$runner = Join-Path $root "scripts\run_ppo_v10_pilot.py"
$cells = foreach ($arm in @("ppo_v10_gate_nav", "ppo_v10_gate_nav_streak")) {
    foreach ($seed in 0..4) {
        [pscustomobject]@{ Arm = $arm; Seed = $seed }
    }
}
$running = @()
foreach ($cell in $cells) {
    while ($running.Count -ge $MaxParallel) {
        $finished = $running | Where-Object { $_.Process.HasExited }
        if (-not $finished) { Start-Sleep -Seconds 2; continue }
        foreach ($item in $finished) {
            if ($item.Process.ExitCode -ne 0) {
                throw "Pilot cell failed: $($item.Arm) seed $($item.Seed); see $($item.Log)"
            }
            $running = @($running | Where-Object { $_.Process.Id -ne $item.Process.Id })
        }
    }
    $logDir = Join-Path $root "artifacts\logs\v10_ppo_pilot_supervisor"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $log = Join-Path $logDir "$($cell.Arm)_seed$($cell.Seed).log"
    $arguments = @($runner, "--run", "--arm", $cell.Arm, "--seed", $cell.Seed,
                   "--threads", $Threads)
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{ Process = $process; Arm = $cell.Arm; Seed = $cell.Seed; Log = $log }
}
foreach ($item in $running) {
    $item.Process.WaitForExit()
    if ($item.Process.ExitCode -ne 0) {
        throw "Pilot cell failed: $($item.Arm) seed $($item.Seed); see $($item.Log)"
    }
}
& $python $runner --aggregate
exit $LASTEXITCODE

