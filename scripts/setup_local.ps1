[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host "Criando ambiente virtual Python em .venv..."
        py -3.12 -m venv .venv
    }

    Write-Host "Instalando PyTorch/Ultralytics e o projeto..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e ".[dev]"

    Write-Host "Verificando a instalacao..."
    & $venvPython -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('Dispositivo: CPU')"
    & $venvPython -m unittest discover -s tests -v
}
finally {
    Pop-Location
}

Write-Host "Instalacao local concluida. Execute: .\scripts\run_local.ps1 -Mode Check -Phase all"
