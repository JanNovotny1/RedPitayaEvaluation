# Creates a Python virtual environment and installs the dependencies.
# Run from the repo root:
#   .\setup_env.ps1                                  # Python 3.13 (py launcher, else default install path)
#   .\setup_env.ps1 -PythonVersion 3.12              # another installed version
#   .\setup_env.ps1 -PythonExe "C:\path\python.exe"  # explicit interpreter
# Requires Git on PATH (pyrpl is installed from GitHub).

param(
    [string]$PythonVersion = "3.13",
    [string]$PythonExe = ""
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

function Test-Python {
    # Returns $true if the interpreter starts and finds its standard library
    param([string]$Exe, [string[]]$PreArgs = @())
    try {
        & $Exe @PreArgs -c "import encodings, sys; print(sys.version); print(sys.prefix)" | Out-Host
        return ($LASTEXITCODE -eq 0)
    } catch {
        Write-Host "  could not run: $($_.Exception.Message)"
        return $false
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git not found on PATH - install Git for Windows first (needed to install pyrpl)."
}

# Resolve the base interpreter: explicit path > default install path > py launcher
$basePy = $null
$basePre = @()
$ver = $PythonVersion -replace '\.', ''
$candidates = @()
if ($PythonExe) { $candidates += [pscustomobject]@{ Exe = $PythonExe; Pre = @() } }
$candidates += [pscustomobject]@{ Exe = "$env:LOCALAPPDATA\Programs\Python\Python$ver\python.exe"; Pre = @() }
$candidates += [pscustomobject]@{ Exe = "$env:ProgramFiles\Python$ver\python.exe"; Pre = @() }
$candidates += [pscustomobject]@{ Exe = "py"; Pre = @("-$PythonVersion") }

foreach ($c in $candidates) {
    $exe = $c.Exe
    $pre = @($c.Pre)
    if ($exe -ne "py" -and -not (Test-Path $exe)) { continue }
    Write-Host "Checking $exe $($pre -join ' ') ..."
    if (Test-Python $exe $pre) { $basePy = $exe; $basePre = $pre; break }
}
if (-not $basePy) {
    throw ("No working Python $PythonVersion found. Install it, or pass the interpreter explicitly: " +
           ".\setup_env.ps1 -PythonExe C:\path\to\python.exe")
}

# Remove a half-created environment from an earlier failed run
if ((Test-Path ".venv") -and -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Removing incomplete .venv ..."
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv with $basePy ..."
    Invoke-Checked $basePy ($basePre + @("-m", "venv", ".venv"))
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements.txt")

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1"
