param([Parameter(Mandatory=$true)][string]$StudentId)
$ErrorActionPreference = 'Stop'
$runtimeRoot = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
$artifactNode = Join-Path $runtimeRoot 'node\bin\node.exe'
$artifactPython = Join-Path $runtimeRoot 'python\python.exe'
$skillRoot = Join-Path $env:USERPROFILE '.codex\plugins\cache\openai-primary-runtime'
while (-not (Test-Path -LiteralPath 'working\experiments_complete.txt')) {
    Start-Sleep -Seconds 15
}
if (Test-Path -LiteralPath 'outputs\report\MAAI901_Technical_Report_Ayman_Lakhnati.pdf') {
    throw 'Report already exists. Review it before an intentional revision.'
}
& $artifactNode (Join-Path $skillRoot 'pdf\26.1020.11127\skills\pdf\container_tools\mark_artifact_operation_started.mjs') --operation-kind create --expected-output-count 1 --output-format pdf
if ($LASTEXITCODE -ne 0) { throw 'PDF operation marker failed' }
& $artifactPython scripts/build_report.py --student-id $StudentId
if ($LASTEXITCODE -ne 0) { throw 'Report build failed; inspect diagnostics before retrying' }
& $artifactPython scripts/render_report.py
if ($LASTEXITCODE -ne 0) { throw 'Report rendering failed' }
& $artifactNode (Join-Path $skillRoot 'presentations\26.1020.11127\skills\presentations\container_tools\mark_artifact_operation_started.mjs') --operation-kind create --expected-output-count 1 --output-format pptx
if ($LASTEXITCODE -ne 0) { throw 'Presentation operation marker failed' }
& $artifactNode scripts/build_presentation.mjs
if ($LASTEXITCODE -ne 0) { throw 'Presentation build failed; inspect diagnostics before retrying' }
'FILES BUILT; VISUAL REVIEW AND CANVA IMPORT STILL REQUIRED' | Set-Content -LiteralPath 'working\artifacts_built.txt'
