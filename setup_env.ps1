# Creates a Python virtual environment and installs the dependencies.
# Run from the repo root:
#   .\setup_env.ps1                     # uses Python 3.12 via the py launcher
#   .\setup_env.ps1 -PythonVersion 3.11 # pick another installed version
# List installed versions with:  py -0p
# Requires Git on PATH (pyrpl is installed from GitHub).

param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$ArgList)
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Exe $($ArgList -join ' ')"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git not found on PATH - install Git for Windows first (needed to install pyrpl)."
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found - install Python $PythonVersion from python.org."
}

# Sanity check: the chosen interpreter must start and find its standard library
Write-Host "Checking Python $PythonVersion ..."
& py "-$PythonVersion" -c "import encodings, sys; print(sys.version); print(sys.prefix)"
if ($LASTEXITCODE -ne 0) {
    throw ("Python $PythonVersion is missing or broken. Run 'py -0p' to list installs, " +
           "repair or install it, or pass another version: .\setup_env.ps1 -PythonVersion 3.11")
}

# Remove a half-created environment from an earlier failed run
if ((Test-Path ".venv") -and -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Removing incomplete .venv ..."
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    Invoke-Checked "py" @("-$PythonVersion", "-m", "venv", ".venv")
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements.txt")

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
