[CmdletBinding()]
param(
    [ValidateSet("Check", "Full")]
    [string]$Mode = "Check",

    [ValidateSet("loss", "search", "promote200", "confirm300", "all", "final-test")]
    [string]$Phase = "all",

    [ValidateRange(0, 8)]
    [int]$RunIndex = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$data = Join-Path $projectRoot "datasets\plantdoc_yolo_27\data.yaml"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente nao instalado. Execute primeiro: .\scripts\setup_local.ps1"
}
if (-not (Test-Path -LiteralPath $data)) {
    throw "Dataset nao encontrado em datasets\plantdoc_yolo_27."
}

if ($Mode -eq "Full") {
    $config = Join-Path $projectRoot "configs\colab_protocol.json"
    $runs = Join-Path $projectRoot "runs\local_full"
    Write-Warning "Modo cientifico completo em CPU: pode levar muitos dias. As execucoes concluidas e checkpoints sao preservados."
}
else {
    $config = Join-Path $projectRoot "configs\local_check.json"
    $runs = Join-Path $projectRoot "runs\local_check"
    Write-Warning "Modo Check valida o pipeline, mas seus resultados NAO devem ser usados como resultados cientificos do TCC."
}

Push-Location $projectRoot
try {
    & $venvPython -m plantdoc_tcc audit --data $data --expected-classes 27
    if ($LASTEXITCODE -ne 0) { throw "A auditoria do dataset falhou." }

    if ($Phase -eq "final-test") {
        if ($Mode -ne "Full") { throw "O teste final so pode ser executado com -Mode Full." }
        & $venvPython -m plantdoc_tcc final-test --data $data --project $runs
    }
    else {
        $arguments = @(
            "-m", "plantdoc_tcc", "staged",
            "--data", $data,
            "--config", $config,
            "--project", $runs,
            "--phase", $Phase,
            "--device", "cpu"
        )
        if ($RunIndex -gt 0) {
            $arguments += @("--run-index", $RunIndex)
        }
        & $venvPython @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "A execucao terminou com erro." }
}
finally {
    Pop-Location
}
