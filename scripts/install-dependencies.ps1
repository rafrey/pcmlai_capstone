#!/usr/bin/env pwsh

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$venvDir = Join-Path $repoRoot '.venv'

if (-not ($env:OS -eq 'Windows_NT')) {
    Write-Error 'scripts/install-dependencies.ps1 is intended for PowerShell (pwsh) on Windows. Use scripts/install-dependencies.sh on Linux, WSL2, or macOS.'
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    Write-Error 'uv is required but was not found on PATH. Install it first, then rerun this script.'
}

Write-Host "Creating or refreshing virtual environment at $venvDir"
& uv venv --python 3.13.7 --allow-existing $venvDir
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$pythonBin = Join-Path $venvDir 'Scripts/python.exe'
if (-not (Test-Path $pythonBin)) {
    Write-Error "Expected Python interpreter was not created at $pythonBin"
}

Write-Host 'Installing project dependencies with uv'
& uv pip install --python $pythonBin -r (Join-Path $repoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host 'Checking TensorFlow import and GPU visibility'
$probeCommand = 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
$probeOutput = & $pythonBin -c $probeCommand 2>&1 | Out-String
$probeExitCode = $LASTEXITCODE
$probeOutput = $probeOutput.Trim()

if ($probeOutput) {
    Write-Host $probeOutput
}

if ($probeExitCode -ne 0) {
    Write-Error 'TensorFlow probe failed. Review the output above and fix the environment before opening the notebook.'
}

$missingLibraryPattern = 'Cannot dlopen some GPU libraries|DLL load failed|cudart|cublas|cudnn|cusolver|cusparse|cufft|curand|nccl|nvrtc|nvJitLink|cupti|tensorflow_framework'
$gpuDevicePattern = 'PhysicalDevice\('

if ($probeOutput -match $gpuDevicePattern) {
    Write-Host 'TensorFlow reported at least one GPU device.'
    exit 0
}

if ($probeOutput -match $missingLibraryPattern) {
    Write-Warning 'TensorFlow reported missing GPU-related libraries.'
}

Write-Host 'TensorFlow imported successfully, but no GPU devices were reported.'
Write-Host 'Native Windows TensorFlow pip environments are CPU-only in current releases.'
Write-Host 'If you need GPU acceleration for this project, use WSL2 with Ubuntu or an Ubuntu VM instead of native Windows.'
exit 0