# Creates a Python virtual environment and installs the dependencies.
# Run from the repo root:  .\setup_env.ps1
# Requires Python 3.10-3.12 and Git on PATH (pyrpl is installed from GitHub).

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git not found on PATH - install Git for Windows first (needed to install pyrpl)."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    py -3 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
