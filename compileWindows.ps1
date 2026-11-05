<#
.SYNOPSIS
  Build this Blender fork on Windows x64 with the full Cycles feature set (CPU, CUDA, OptiX)
  and install it to .\install (blender.exe with all of its resources).

.DESCRIPTION
  Configures Blender's "release" feature set (build_files\cmake\config\blender_release.cmake)
  with CMake + Ninja inside a Visual Studio developer environment, then builds and installs it.

  It detects and wires up, without any manual setup:
    * Visual Studio 2022 or newer (vswhere) and its bundled Ninja and CMake.
    * The newest installed CUDA toolkit (CUDA_PATH_V*), for the CUDA kernel binaries and the
      OptiX kernels.
    * The newest OptiX SDK 8.0 or newer (OPTIX_ROOT_DIR or C:\ProgramData\NVIDIA Corporation).
    * The GPUs in this machine (nvidia-smi), so by default CUDA binaries are only compiled for the
      architectures that are present. Use -CudaArch release for a redistributable build.
    * Blender's precompiled libraries (lib\windows_x64), fetched on first use. The LLVM debug
      libraries (3.7 GB) are skipped unless -FullLibraries is given, a Release build does not
      use them.

  AMD HIP and Intel oneAPI GPU binaries are built only when their SDKs are installed (-WithHIP,
  -WithOneAPI).

.EXAMPLE
  .\compileWindows.ps1
.EXAMPLE
  .\compileWindows.ps1 -CudaArch release -Package
.EXAMPLE
  .\compileWindows.ps1 -NoCuda -Jobs 8
.EXAMPLE
  .\compileWindows.ps1 -KernelJobs 1
.EXAMPLE
  .\compileWindows.ps1 -Test
#>
[CmdletBinding()]
param(
  # CMake build directory. Default: .\build\windows_x64_<BuildType>
  [string]$BuildDir = "",
  # Install directory (blender.exe and resources). Default: .\install
  [string]$ReleaseDir = "",
  # Release (default) or RelWithDebInfo. Debug needs -FullLibraries.
  [ValidateSet("Release", "RelWithDebInfo", "Debug")]
  [string]$BuildType = "Release",
  # Parallel jobs. Default: number of logical processors.
  [int]$Jobs = 0,
  # Cycles kernels (CPU kernel files, CUDA and OptiX modules) compiled at once. Each needs up to
  # ~16 GB of memory. Default (0): one per 32 GB of physical memory, at least one.
  [int]$KernelJobs = 0,
  # CUDA architectures: "auto" (GPUs in this machine), "release" (Blender's release list),
  # or an explicit list such as "sm_86;sm_89;compute_75".
  [string]$CudaArch = "auto",
  # Build without CUDA kernel binaries and without OptiX.
  [switch]$NoCuda,
  # Build without OptiX.
  [switch]$NoOptix,
  # Path to the OptiX SDK (default: OPTIX_ROOT_DIR or newest under C:\ProgramData\NVIDIA Corporation).
  [string]$OptixRoot = "",
  # Path to the CUDA toolkit (default: newest CUDA_PATH_V* or CUDA_PATH).
  [string]$CudaRoot = "",
  # Build AMD HIP kernel binaries (needs the HIP SDK, HIP_PATH).
  [switch]$WithHIP,
  # Build the Intel oneAPI device and binaries (needs the full lib\windows_x64\dpcpp).
  [switch]$WithOneAPI,
  # Fetch all library files, including the LLVM debug libraries.
  [switch]$FullLibraries,
  # Fail instead of fetching when libraries are missing.
  [switch]$NoFetchLibraries,
  # Remove the install directory before installing.
  [switch]$CleanRelease,
  # Reconfigure from scratch (cmake --fresh).
  [switch]$Fresh,
  # Also create a zip of the install directory in .\build.
  [switch]$Package,
  # Skip the final checks with the installed blender.exe.
  [switch]$NoSmokeTest,
  # After building, run the feature demo/parity test (tests\python\jcycles_feature_demo.py).
  [switch]$Test,
  # Launch the built Blender afterwards.
  [switch]$Run,
  # Launch the built Blender with this .blend file afterwards.
  [string]$Open = "",
  # Extra CMake arguments, e.g. @("-DWITH_TRACY=OFF").
  [string[]]$CMakeArgs = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3

function Log([string]$Message) { Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan }
function Warn([string]$Message) { Write-Host "warning: $Message" -ForegroundColor Yellow }
function Die([string]$Message) { Write-Host ""; Write-Host "error: $Message" -ForegroundColor Red; exit 1 }

function Invoke-Checked([string]$File, [string[]]$Arguments) {
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) { Die "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')" }
}

$RootDir = $PSScriptRoot
if (-not $RootDir) { $RootDir = (Get-Location).Path }

if ($env:OS -ne "Windows_NT") { Die "compileWindows.ps1 builds on Windows. On Linux run ./compileLinux.sh, on macOS ./compile.sh." }
if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") { Die "compileWindows.ps1 builds Windows x64 (this machine: $($env:PROCESSOR_ARCHITECTURE))." }

$Arch = "x64"
if (-not $BuildDir) { $BuildDir = Join-Path $RootDir "build\windows_${Arch}_$BuildType" }
if (-not $ReleaseDir) { $ReleaseDir = Join-Path $RootDir "install" }
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$ReleaseDir = [System.IO.Path]::GetFullPath($ReleaseDir)
$LibDir = Join-Path $RootDir "lib\windows_$Arch"
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
if ($KernelJobs -lt 0) { Die "-KernelJobs must be 0 (automatic) or a positive number." }
if ($BuildType -eq "Debug") { $FullLibraries = $true }

foreach ($dir in @($RootDir, $BuildDir)) {
  if ($dir.Contains(" ")) { Die "Paths with spaces are not supported by the CUDA/OptiX kernel build: $dir" }
}
# MSVC does not support paths over 260 characters. The longest object paths add about 150
# characters to the build directory.
if ($BuildDir.Length -gt 105) {
  Warn "The build directory path is $($BuildDir.Length) characters long. Object paths may exceed MSVC's 260 character limit; use a shorter -BuildDir if compilation fails with error C1083."
}

# ------------------------------------------------------------------------------------------------
# Visual Studio developer environment.

function Find-VisualStudio {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { return $null }
  $path = & $vswhere -latest -products * -version "[17.0," `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if ($LASTEXITCODE -ne 0 -or -not $path) { return $null }
  return ($path | Select-Object -First 1)
}

function Import-VsEnvironment([string]$VsPath) {
  $vcvars = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"
  if (-not (Test-Path $vcvars)) { Die "vcvars64.bat not found in $VsPath" }
  $lines = & cmd.exe /d /c "`"$vcvars`" >nul 2>&1 && set"
  if ($LASTEXITCODE -ne 0) { Die "Failed to set up the Visual Studio environment ($vcvars)." }
  foreach ($line in $lines) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
  }
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $vs = Find-VisualStudio
  if (-not $vs) { Die "Visual Studio 2022 (or newer) with the 'Desktop development with C++' workload is required." }
  Log "Setting up the Visual Studio environment ($vs)"
  Import-VsEnvironment $vs
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { Die "cl.exe is not available after loading the Visual Studio environment." }

$Ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
if (-not $Ninja) { Die "ninja.exe was not found. Install the 'C++ CMake tools for Windows' Visual Studio component or put ninja in PATH." }
$CMake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
if (-not $CMake) { Die "cmake.exe was not found." }

# ------------------------------------------------------------------------------------------------
# Libraries and Git LFS data.

function Test-LibrariesReady {
  if (-not (Test-Path (Join-Path $LibDir ".git"))) { return $false }
  foreach ($d in @("python", "ffmpeg", "OpenImageIO", "usd", "embree", "osl", "openimagedenoise")) {
    if (-not (Test-Path (Join-Path $LibDir $d))) { return $false }
  }
  # An LFS pointer instead of the real file means the LFS content was not fetched.
  $probe = Join-Path $LibDir "embree\bin\embree4.dll"
  if ((Test-Path $probe) -and ((Get-Item $probe).Length -lt 4096)) { return $false }
  return $true
}

function Test-LfsReady {
  $startup = Join-Path $RootDir "release\datafiles\startup.blend"
  if (-not (Test-Path $startup)) { return $true }
  return ((Get-Item $startup).Length -ge 1024)
}

# Blender's upstream LFS server holds every object; forks often lack some of them.
$UpstreamLfs = "https://projects.blender.org/blender/blender.git/info/lfs"

if (-not (Test-LibrariesReady)) {
  if ($NoFetchLibraries) { Die "Libraries are missing from $LibDir. Run without -NoFetchLibraries to fetch them." }
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git is required to fetch libraries." }
  & git lfs version *> $null
  if ($LASTEXITCODE -ne 0) { Die "Git LFS is required to fetch Blender's precompiled libraries." }
  Log "Fetching Blender's Windows $Arch libraries into $LibDir"
  Push-Location $RootDir
  try {
    Invoke-Checked "git" @("lfs", "install", "--skip-repo")
    Invoke-Checked "git" @("config", "submodule.lib/windows_$Arch.update", "checkout")
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    try { Invoke-Checked "git" @("submodule", "update", "--init", "lib/windows_$Arch") }
    finally { Remove-Item env:GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue }
  } finally { Pop-Location }
  Push-Location $LibDir
  try {
    if ($FullLibraries) { & git config --unset lfs.fetchexclude 2>$null; $global:LASTEXITCODE = 0 }
    else { Invoke-Checked "git" @("config", "lfs.fetchexclude", "llvm/debug/**") }
    Invoke-Checked "git" @("lfs", "pull")
  } finally { Pop-Location }
  if (-not (Test-LibrariesReady)) { Die "Libraries are still incomplete after fetching: $LibDir" }
}

if (-not (Test-LfsReady)) {
  if ($NoFetchLibraries) { Die "Blender's Git LFS data is not fetched (release\datafiles\startup.blend is a pointer). Run: git lfs pull" }
  Log "Fetching Blender's Git LFS data (without tests\files)"
  Push-Location $RootDir
  try { Invoke-Checked "git" @("-c", "lfs.fetchexclude=tests/files/**", "-c", "lfs.url=$UpstreamLfs", "lfs", "pull") }
  finally { Pop-Location }
  if (-not (Test-LfsReady)) { Die "Blender's Git LFS data is still incomplete after git lfs pull." }
}

# ------------------------------------------------------------------------------------------------
# CUDA, OptiX and GPU architectures.

function Get-NvccVersion([string]$Nvcc) {
  $out = & $Nvcc --version 2>$null | Out-String
  if ($out -match "release (\d+)\.(\d+)") { return [version]"$($Matches[1]).$($Matches[2])" }
  return $null
}

function Find-CudaToolkit {
  $roots = @()
  if ($CudaRoot) { $roots += $CudaRoot }
  else {
    foreach ($item in Get-ChildItem env: | Where-Object { $_.Name -match '^CUDA_PATH_V\d+_\d+$' }) { $roots += $item.Value }
    if ($env:CUDA_PATH) { $roots += $env:CUDA_PATH }
    $default = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $default) { $roots += (Get-ChildItem $default -Directory | ForEach-Object { $_.FullName }) }
  }
  $best = $null
  foreach ($root in ($roots | Select-Object -Unique)) {
    $nvcc = Join-Path $root "bin\nvcc.exe"
    if (-not (Test-Path $nvcc)) { continue }
    $ver = Get-NvccVersion $nvcc
    if (-not $ver) { continue }
    if (-not $best -or $ver -gt $best.Version) { $best = [pscustomobject]@{ Root = $root; Nvcc = $nvcc; Version = $ver } }
  }
  return $best
}

function Find-Optix {
  $candidates = @()
  if ($OptixRoot) { $candidates += $OptixRoot }
  elseif ($env:OPTIX_ROOT_DIR) { $candidates += $env:OPTIX_ROOT_DIR }
  else {
    $base = Join-Path $env:ProgramData "NVIDIA Corporation"
    if (Test-Path $base) {
      $candidates += (Get-ChildItem $base -Directory -Filter "OptiX SDK *" | ForEach-Object { $_.FullName })
    }
  }
  $best = $null
  foreach ($c in $candidates) {
    $header = Join-Path $c "include\optix.h"
    if (-not (Test-Path $header)) { continue }
    $line = Select-String -Path $header -Pattern '^#define OPTIX_VERSION\s+(\d+)' | Select-Object -First 1
    if (-not $line) { continue }
    $code = [int]$line.Matches[0].Groups[1].Value
    $ver = [version]("{0}.{1}.{2}" -f [math]::Floor($code / 10000), [math]::Floor(($code % 10000) / 100), ($code % 100))
    if ($ver -lt [version]"8.0.0") { continue }
    if (-not $best -or $ver -gt $best.Version) { $best = [pscustomobject]@{ Root = $c; Version = $ver } }
  }
  return $best
}

function Get-LocalCudaArchs {
  $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  if (-not $smi) { return @() }
  $caps = & $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null
  if ($LASTEXITCODE -ne 0) { return @() }
  $archs = @()
  foreach ($cap in $caps) {
    if ($cap -match '^\s*(\d+)\.(\d+)\s*$') { $archs += "sm_$($Matches[1])$($Matches[2])" }
  }
  return ($archs | Select-Object -Unique)
}

$Cuda = $null
$Optix = $null
$CudaArchList = ""
if (-not $NoCuda) {
  $Cuda = Find-CudaToolkit
  if (-not $Cuda) {
    Warn "No CUDA toolkit found: building without CUDA kernel binaries and OptiX (install CUDA 12 or 13, or pass -NoCuda)."
  } else {
    if ($Cuda.Version.Major -lt 12) {
      Warn "CUDA $($Cuda.Version) is older than the CUDA 12/13 Blender is tested with. It works for sm_50..sm_90, newer GPUs need CUDA 12.8+."
    }
    switch ($CudaArch) {
      "auto" {
        $local = @(Get-LocalCudaArchs)
        if ($local.Count -eq 0) {
          Warn "No NVIDIA GPU detected, compiling Blender's release CUDA architecture list."
          $CudaArchList = ""
        } else { $CudaArchList = ($local -join ";") }
      }
      "release" { $CudaArchList = "" }
      default { $CudaArchList = $CudaArch }
    }
    if (-not $NoOptix) {
      $Optix = Find-Optix
      if (-not $Optix) { Warn "No OptiX SDK 8.0+ found (set OPTIX_ROOT_DIR or pass -OptixRoot): building without OptiX." }
    }
  }
}

$HipPath = $null
if ($WithHIP) {
  if ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin\hipcc.bin.exe"))) { $HipPath = $env:HIP_PATH }
  elseif ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin"))) { $HipPath = $env:HIP_PATH }
  else { Die "-WithHIP needs the AMD HIP SDK (HIP_PATH is not set)." }
}

# ------------------------------------------------------------------------------------------------
# Configure.

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$InitialCache = Join-Path $BuildDir "jcycles_windows_config.cmake"
$cmakeRoot = $RootDir.Replace('\', '/')
$cache = @(
  "# Generated by compileWindows.ps1. Blender's release feature set, adapted to this machine.",
  "include(`"$cmakeRoot/build_files/cmake/config/blender_release.cmake`")",
  "set(WITH_CYCLES_DEVICE_CUDA ON CACHE BOOL `"`" FORCE)",
  "set(WITH_CYCLES_DEVICE_HIP ON CACHE BOOL `"`" FORCE)",
  "# Offline C++ compilation of the GLSL sources is a developer check only. Its generated object",
  "# files have names too long for MSVC's 260 character path limit in deep checkouts.",
  "set(WITH_GPU_SHADER_CPP_COMPILATION OFF CACHE BOOL `"`" FORCE)"
)
if ($Cuda) {
  $cudaRootCMake = $Cuda.Root.Replace('\', '/')
  $cache += "set(WITH_CYCLES_CUDA_BINARIES ON CACHE BOOL `"`" FORCE)"
  $cache += "set(CUDA_TOOLKIT_ROOT_DIR `"$cudaRootCMake`" CACHE PATH `"`" FORCE)"
  $cache += "set(CUDA_NVCC_EXECUTABLE `"$cudaRootCMake/bin/nvcc.exe`" CACHE FILEPATH `"`" FORCE)"
  if ($CudaArchList) { $cache += "set(CYCLES_CUDA_BINARIES_ARCH `"$CudaArchList`" CACHE STRING `"`" FORCE)" }
} else {
  $cache += "set(WITH_CYCLES_CUDA_BINARIES OFF CACHE BOOL `"`" FORCE)"
}
if ($Optix) {
  $cache += "set(WITH_CYCLES_DEVICE_OPTIX ON CACHE BOOL `"`" FORCE)"
  $cache += "set(OPTIX_ROOT_DIR `"$($Optix.Root.Replace('\', '/'))`" CACHE PATH `"`" FORCE)"
} else {
  $cache += "set(WITH_CYCLES_DEVICE_OPTIX OFF CACHE BOOL `"`" FORCE)"
}
if ($HipPath) {
  $cache += "set(WITH_CYCLES_HIP_BINARIES ON CACHE BOOL `"`" FORCE)"
} else {
  $cache += "set(WITH_CYCLES_HIP_BINARIES OFF CACHE BOOL `"`" FORCE)"
  $cache += "set(WITH_CYCLES_DEVICE_HIPRT OFF CACHE BOOL `"`" FORCE)"
}
if ($WithOneAPI) {
  $cache += "set(WITH_CYCLES_DEVICE_ONEAPI ON CACHE BOOL `"`" FORCE)"
  $cache += "set(WITH_CYCLES_ONEAPI_BINARIES ON CACHE BOOL `"`" FORCE)"
} else {
  $cache += "set(WITH_CYCLES_DEVICE_ONEAPI OFF CACHE BOOL `"`" FORCE)"
  $cache += "set(WITH_CYCLES_ONEAPI_BINARIES OFF CACHE BOOL `"`" FORCE)"
}
Set-Content -Path $InitialCache -Value $cache -Encoding ascii

Log "Build configuration"
Write-Host "  Source:       $RootDir"
Write-Host "  Build dir:    $BuildDir ($BuildType, Ninja, $Jobs jobs)"
Write-Host "  Kernel jobs:  $(if ($KernelJobs -gt 0) { $KernelJobs } else { 'automatic, from the physical memory' })"
Write-Host "  Install dir:  $ReleaseDir"
Write-Host "  Libraries:    $LibDir"
Write-Host "  Compiler:     $((Get-Command cl.exe).Source)"
if ($Cuda) {
  $archText = if ($CudaArchList) { $CudaArchList } else { "Blender release list" }
  Write-Host "  CUDA:         $($Cuda.Version) ($($Cuda.Root)), architectures: $archText"
} else { Write-Host "  CUDA:         off" }
if ($Optix) { Write-Host "  OptiX:        $($Optix.Version) ($($Optix.Root))" } else { Write-Host "  OptiX:        off" }
Write-Host "  HIP binaries: $(if ($HipPath) { $HipPath } else { 'off' })"
Write-Host "  oneAPI:       $(if ($WithOneAPI) { 'on' } else { 'off' })"

if ($CleanRelease) {
  if ($ReleaseDir -eq $RootDir -or $ReleaseDir -eq $BuildDir -or $ReleaseDir.Length -le 3) {
    Die "Refusing to remove unsafe install directory: $ReleaseDir"
  }
  if (Test-Path $ReleaseDir) {
    Log "Removing $ReleaseDir"
    Remove-Item -Recurse -Force $ReleaseDir
  }
}

$configure = @(
  "-S", $RootDir,
  "-B", $BuildDir,
  "-G", "Ninja",
  "-C", $InitialCache,
  "-DCMAKE_BUILD_TYPE=$BuildType",
  "-DCMAKE_INSTALL_PREFIX=$ReleaseDir",
  "-DWITH_NINJA_POOL_JOBS=ON",
  "-DCYCLES_KERNEL_MAX_PARALLEL_JOBS=$KernelJobs",
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)
if ($Fresh) { $configure += "--fresh" }
$configure += $CMakeArgs

Log "Configuring"
Invoke-Checked $CMake.Source $configure

Log "Building and installing"
$sw = [Diagnostics.Stopwatch]::StartNew()
Invoke-Checked $CMake.Source @("--build", $BuildDir, "--target", "install", "--parallel", "$Jobs")
Write-Host ("Build and install took {0:hh\:mm\:ss}" -f $sw.Elapsed)

# ------------------------------------------------------------------------------------------------
# Verify the installation.

$BlenderExe = Join-Path $ReleaseDir "blender.exe"
if (-not (Test-Path $BlenderExe)) { Die "blender.exe was not installed: $BlenderExe" }

Log "Verifying the installed Cycles add-on and kernels"
$addon = Get-ChildItem -Path $ReleaseDir -Directory -Recurse -Depth 3 -Filter cycles -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "scripts\\addons_core\\cycles$" } | Select-Object -First 1
if (-not $addon) { Die "The installed Cycles add-on was not found in $ReleaseDir." }
foreach ($name in @("__init__.py", "engine.py", "properties.py", "ui.py")) {
  $src = Join-Path $RootDir "intern\cycles\blender\addon\$name"
  $dst = Join-Path $addon.FullName $name
  if (-not (Test-Path $dst)) { Die "Installed Cycles add-on file is missing: $dst" }
  if ((Get-FileHash $src).Hash -ne (Get-FileHash $dst).Hash) { Die "Installed Cycles add-on file is stale: $name" }
}
$kernelDir = Join-Path $addon.FullName "lib"
$expected = @()
if ($Cuda) {
  if ($CudaArchList) {
    foreach ($a in ($CudaArchList -split ";")) {
      if ($a -match '^sm_') { $expected += "kernel_$a.cubin.zst" } elseif ($a -match '^compute_') { $expected += "kernel_$a.ptx.zst" }
    }
  }
}
if ($Optix) { $expected += "kernel_optix.ptx.zst"; $expected += "kernel_optix_shader_raytrace.ptx.zst" }
foreach ($k in $expected) {
  if (-not (Test-Path (Join-Path $kernelDir $k))) { Die "GPU kernel missing from the installation: $kernelDir\$k" }
}
if ($expected.Count -gt 0) { Write-Host "  GPU kernels: $($expected -join ', ')" }

if (-not $NoSmokeTest) {
  Log "Smoke test"
  $probe = Join-Path $RootDir "tests\python\jcycles_device_probe.py"
  & $BlenderExe --background --factory-startup --python $probe
  if ($LASTEXITCODE -ne 0) { Die "Smoke test failed: $BlenderExe --background --python $probe" }
}

if ($Package) {
  $version = (Get-ChildItem $ReleaseDir -Directory | Where-Object { $_.Name -match '^\d+\.\d+$' } | Select-Object -First 1).Name
  $zip = Join-Path $RootDir "build\blender-jcycles-$version-windows-x64.zip"
  Log "Packaging $zip"
  if (Test-Path $zip) { Remove-Item -Force $zip }
  Compress-Archive -Path (Join-Path $ReleaseDir "*") -DestinationPath $zip -CompressionLevel Optimal
}

if ($Test) {
  Log "Running the feature demo and parity test"
  $demo = Join-Path $RootDir "tests\python\jcycles_feature_demo.py"
  & $BlenderExe --background --factory-startup --python $demo -- --output (Join-Path $RootDir "build\jcycles_feature_demo")
  if ($LASTEXITCODE -ne 0) { Die "Feature demo test failed." }
}

Log "Done"
Write-Host "Blender: $BlenderExe"
Write-Host "JCycles add-on for an official Blender: .\compileCycles.ps1"

if ($Open) {
  if (-not (Test-Path $Open)) { Die "Blend file does not exist: $Open" }
  Start-Process -FilePath $BlenderExe -ArgumentList @("`"$Open`"")
} elseif ($Run) {
  Start-Process -FilePath $BlenderExe
}
