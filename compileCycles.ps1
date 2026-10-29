<#
.SYNOPSIS
  Package this repository's Blender build as the JCycles add-on (Windows).

.DESCRIPTION
  Run .\compile.ps1 first: it builds and installs Blender to .\install. This script then writes
  .\build\JCycles, a folder you can zip and install in an official Blender as an add-on. It adds
  a "JCycles" render engine that renders with this repository's Cycles (the build is bundled
  inside). On macOS and Linux use ./compileCycles.sh.

.EXAMPLE
  .\compileCycles.ps1
.EXAMPLE
  .\compileCycles.ps1 -Compile -Zip
#>
[CmdletBinding()]
param(
  # Run .\compile.ps1 first (incremental build and install).
  [switch]$Compile,
  # Installed build to package. Default: .\install
  [string]$Runtime = "",
  # Add-on folder to write. Default: .\build\JCycles
  [string]$Output = "",
  # Also write .\build\JCycles-<version>-<platform>.zip
  [switch]$Zip,
  # Package even if the installed build is older than the sources.
  [switch]$AllowStale,
  # Arguments for compile.ps1 when -Compile is used, e.g. @("-Ninja").
  [string[]]$CompileArgs = @()
)

$ErrorActionPreference = "Stop"
$RootDir = $PSScriptRoot
if (-not $RootDir) { $RootDir = (Get-Location).Path }

if ($env:OS -ne "Windows_NT") {
  Write-Host "error: on macOS and Linux run ./compileCycles.sh" -ForegroundColor Red
  exit 1
}

if ($Compile) {
  & (Join-Path $RootDir "compile.ps1") @CompileArgs
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$python = $null
$pythonArgs = @()
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
$libPython = Join-Path $RootDir "lib\windows_$arch\python"
if (Test-Path $libPython) {
  $found = Get-ChildItem -Path $libPython -Recurse -Filter python.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "_d\.exe$" } | Select-Object -First 1
  if ($found) { $python = $found.FullName }
}
if (-not $python -and (Get-Command py -ErrorAction SilentlyContinue)) { $python = "py"; $pythonArgs = @("-3") }
if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) { $python = "python" }
if (-not $python) {
  Write-Host "error: Python 3.8 or newer is required (lib\windows_$arch\python, the py launcher or python)" -ForegroundColor Red
  exit 1
}

$arguments = $pythonArgs + @((Join-Path $RootDir "build_files\jcycles\build_jcycles.py"))
if ($Runtime) { $arguments += @("--runtime", $Runtime) }
if ($Output) { $arguments += @("--output", $Output) }
if ($Zip) { $arguments += "--zip" }
if ($AllowStale) { $arguments += "--allow-stale" }

& $python @arguments
exit $LASTEXITCODE
