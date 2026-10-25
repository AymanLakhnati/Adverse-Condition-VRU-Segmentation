$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ExperimentExecutionState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$null = [ExperimentExecutionState]::SetThreadExecutionState([uint32]2147483649)
$taskPython = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
try {
    while (-not (Test-Path -LiteralPath 'data\acdc\download_provenance.json')) {
        Start-Sleep -Seconds 15
    }
    do {
        Start-Sleep -Seconds 15
        $primaryDone = $false
        if (Test-Path -LiteralPath 'results\inference_run.json') {
            try { $primaryDone = (Get-Content -Raw -LiteralPath 'results\inference_run.json' | ConvertFrom-Json).status -eq 'completed' } catch {}
        }
    } while (-not $primaryDone)
    & $taskPython -u -m roadseg.metrics results/manifest.csv --output results --bootstrap 2000 --require-official-validation --path-root .
    if ($LASTEXITCODE -ne 0) { throw 'Primary evaluation failed' }
    & $taskPython -u -m roadseg.tune_dinov3 --device cuda --refit
    if ($LASTEXITCODE -ne 0) { throw 'DINOv3 tuning or training failed' }
    & $taskPython -u -m roadseg.evaluate_dinov3 --device cuda --batch-size 4
    if ($LASTEXITCODE -ne 0) { throw 'DINOv3 evaluation failed' }
    & $taskPython -u -m roadseg.visualize
    if ($LASTEXITCODE -ne 0) { throw 'Visualization failed' }
    'ALL EXPERIMENTS AND FIGURES COMPLETED' | Set-Content -LiteralPath 'working\experiments_complete.txt'
} finally {
    $null = [ExperimentExecutionState]::SetThreadExecutionState([uint32]2147483648)
}
