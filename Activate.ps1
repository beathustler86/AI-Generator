# Activate.ps1 - Environment + launcher
[CmdletBinding()]
param(
    [switch]$Reinstall,          # Force reinstall core deps (excluding torch unless missing)
    [switch]$NoInstall,          # Skip all dependency installs
    [switch]$Dev,                # Enable dev dependency checking inside ai_start.py
    [switch]$AiStart = $true,    # Use ai_start.py (recommended). Use -AiStart:$false to run a raw script.
    [string]$Script = "src\launch_gui.py",
    [string]$PythonVersion = "3.10",
    [switch]$VerboseDeps         # Print per-module checks
)

$ErrorActionPreference = "Stop"

$Root  = Split-Path -Parent $PSCommandPath
$Venv  = Join-Path $Root ".venv"
$PyExe = Join-Path $Venv "Scripts\python.exe"
$CudaIndex = "https://download.pytorch.org/whl/cu124"

# Torch versions (with +cu124 tag)
$TorchSpec = @(
    "torch==2.6.0+cu124",
    "torchvision==0.21.0+cu124",
    "torchaudio==2.6.0+cu124"
)

# Core deps (PySide6 added)
$CoreDeps = @(
    "PySide6",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pillow"
)

Write-Host "== AI Generator Launcher ==" -ForegroundColor Cyan
Write-Host "[Root]        $Root"
Write-Host "[Python ver]  $PythonVersion"
Write-Host "[Mode]        Reinstall=$($Reinstall.IsPresent) NoInstall=$($NoInstall.IsPresent) Dev=$($Dev.IsPresent) AiStart=$($AiStart.IsPresent)"

# --- Ensure venv ---
if (-not (Test-Path $PyExe)) {
    Write-Host "[VENV] Creating virtual environment..." -ForegroundColor Yellow
    py -$PythonVersion -m venv $Venv
}
if (-not (Test-Path $PyExe)) {
    Write-Host "[ERROR] Venv python missing after creation." -ForegroundColor Red
    exit 1
}

# --- Helpers ---
function Test-Mod {
    param([string]$mod)
    & $PyExe - <<PY
import importlib,sys
sys.exit(0 if importlib.util.find_spec("$mod") else 1)
PY
    return ($LASTEXITCODE -eq 0)
}

function Ensure-Torch {
    $missing = @()
    foreach ($spec in $TorchSpec) {
        $name = $spec.Split('==')[0]
        if (-not (Test-Mod $name)) { $missing += $spec }
    }
    if ($missing.Count -gt 0 -or $Reinstall) {
        Write-Host "[PIP] Installing Torch stack (CUDA 12.4)..." -ForegroundColor Green
        & $PyExe -m pip install --extra-index-url $CudaIndex @TorchSpec
    } else {
        Write-Host "[PIP] Torch OK" -ForegroundColor DarkGreen
    }
}

function Ensure-Core {
    $need = @()
    if ($Reinstall) {
        $need = $CoreDeps
    } else {
        foreach ($d in $CoreDeps) {
            $ok = Test-Mod $d
            if (-not $ok) { $need += $d }
            if ($VerboseDeps) {
                Write-Host ("[Check] {0,-12} {1}" -f $d, ($ok ? "OK" : "MISSING"))
            }
        }
    }
    if ($need.Count -gt 0) {
        Write-Host "[PIP] Installing core: $($need -join ', ')" -ForegroundColor Green
        & $PyExe -m pip install @need
    } else {
        Write-Host "[PIP] Core deps OK" -ForegroundColor DarkGreen
    }
}

# --- Dependency phase (optional) ---
if (-not $NoInstall) {
    Write-Host "[PIP] Upgrading pip (quiet)" -ForegroundColor Yellow
    & $PyExe -m pip install --disable-pip-version-check --upgrade pip | Out-Null
    Ensure-Torch
    Ensure-Core
} else {
    Write-Host "[PIP] Skipped installs (--NoInstall)" -ForegroundColor Yellow
}

# --- Dev flags exported for ai_start.py ---
if ($Dev) {
    $env:DEV_CHECK_DEPS = "1"
    Write-Host "[Env] DEV_CHECK_DEPS=1" -ForegroundColor Cyan
} else {
    Remove-Item Env:DEV_CHECK_DEPS -ErrorAction SilentlyContinue
}

# --- Run target ---
if ($AiStart) {
    Write-Host "[RUN] python -m src.ai_start" -ForegroundColor Cyan
    & $PyExe -m src.ai_start
} else {
    Write-Host "[RUN] $Script" -ForegroundColor Cyan
    & $PyExe $Script
}

$exitCode = $LASTEXITCODE
Write-Host "[EXIT] Code $exitCode" -ForegroundColor Cyan
exit $exitCode
