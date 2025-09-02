@echo off
setlocal ENABLEDELAYEDEXPANSION

rem ------------------------------------------------------------------
rem cd to script directory (project root assumed here)
rem ------------------------------------------------------------------
pushd "%~dp0"

rem ------------------------------------------------------------------
rem Ensure virtual environment exists (optional bootstrap)
rem ------------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment (.venv)...
  py -3 -m venv .venv || python -m venv .venv
)

rem ------------------------------------------------------------------
rem Activate venv (cmd shell)
rem ------------------------------------------------------------------
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  echo [WARN] Could not activate venv (.venv\Scripts\activate.bat missing)
)

rem ------------------------------------------------------------------
rem Ensure default env vars are set
rem ------------------------------------------------------------------
if not defined ENABLE_XFORMERS set ENABLE_XFORMERS=0
if not defined ENABLE_PROMPT_CACHE set ENABLE_PROMPT_CACHE=1
if not defined MAX_PROMPT_CACHE set MAX_PROMPT_CACHE=64
if not defined DEFAULT_SCHEDULER set DEFAULT_SCHEDULER=dpmpp_2m_karras
if not defined ADAPT_STEPS set ADAPT_STEPS=1
if not defined ENABLE_BFLOAT16 set ENABLE_BFLOAT16=1
if not defined ENABLE_TF32 set ENABLE_TF32=1
if not defined PYTORCH_CUDA_ALLOC_CONF set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if not defined WARMUP_ENABLE set WARMUP_ENABLE=1
if not defined ENABLE_VAE_SLICING set ENABLE_VAE_SLICING=0

rem ------------------------------------------------------------------
rem Guarantee pipeline cache dir + PYTHONPATH
rem ------------------------------------------------------------------
if not defined PIPELINE_CACHE_DIR set "PIPELINE_CACHE_DIR=%CD%\src\pipeline_cache"
if not exist "%PIPELINE_CACHE_DIR%" mkdir "%PIPELINE_CACHE_DIR%"
set "PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%"

rem ------------------------------------------------------------------
rem Quick diagnostics
rem ------------------------------------------------------------------
echo [DIAG] Python: %PYTHON%
python -c "import sys;print('[DIAG] sys.executable=',sys.executable);print('[DIAG] sys.path[0]=',sys.path[0])"
python -c "import torch,importlib;print('[DIAG] torch',torch.__version__);print('[DIAG] xformers?',bool(importlib.util.find_spec('xformers')))"

rem ------------------------------------------------------------------
rem Launch GUI as module (package-safe)
rem ------------------------------------------------------------------
python -m src.launch_gui %*

set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
