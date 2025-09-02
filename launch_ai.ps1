param(
  [string[]]$Args
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Write-Host "[INFO] Creating venv..."
  py -3 -m venv .venv
}

& "$Root\.venv\Scripts\Activate.ps1"

if (-not $env:PIPELINE_CACHE_DIR) {
  $env:PIPELINE_CACHE_DIR = Join-Path $Root "src\pipeline_cache"
  if (-not (Test-Path $env:PIPELINE_CACHE_DIR)) { New-Item -ItemType Directory -Path $env:PIPELINE_CACHE_DIR | Out-Null }
}

$env:PYTHONPATH = "$Root;$Root\src;$env:PYTHONPATH"

if (-not $env:ENABLE_XFORMERS) { $env:ENABLE_XFORMERS = "0" }
if (-not $env:ENABLE_PROMPT_CACHE) { $env:ENABLE_PROMPT_CACHE = "1" }
if (-not $env:MAX_PROMPT_CACHE) { $env:MAX_PROMPT_CACHE = "64" }
if (-not $env:DEFAULT_SCHEDULER) { $env:DEFAULT_SCHEDULER = "dpmpp_2m_karras" }
if (-not $env:ADAPT_STEPS) { $env:ADAPT_STEPS = "1" }
if (-not $env:ENABLE_BFLOAT16) { $env:ENABLE_BFLOAT16 = "1" }
if (-not $env:ENABLE_TF32) { $env:ENABLE_TF32 = "1" }
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True" }
if (-not $env:WARMUP_ENABLE) { $env:WARMUP_ENABLE = "1" }

Write-Host "[DIAG] Python:" (Get-Command python).Source
python -c "import torch,importlib,sys;print('torch',torch.__version__,'xformers?',bool(importlib.util.find_spec('xformers')), 'exe', sys.executable)"
python -m src.launch_gui @Args
$ec = $LASTEXITCODE
Pop-Location
exit $ec
