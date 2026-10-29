<#
.SYNOPSIS
  Build Blender on Windows and install the complete build to .\install (blender.exe and its resources).

.DESCRIPTION
  Windows counterpart of compile.sh (macOS and Linux). It drives Blender's own make.bat with the
  "release" feature set (build_files\cmake\config\blender_release.cmake), so Visual Studio
  detection, library checks and CMake configuration are the ones the Blender project maintains.
  The installed build is then copied to the release directory.

.EXAMPLE
  .\compile.ps1
.EXAMPLE
  .\compile.ps1 -Ninja -Run
.EXAMPLE
  .\compile.ps1 -BuildDir D:\b\blender -ReleaseDir D:\b\install
#>
[CmdletBinding()]
param(
  # CMake build directory. Default: .\build\windows_<arch>_Release
  [string]$BuildDir = "",
  # Where the finished build is copied. Default: .\install
  [string]$ReleaseDir = "",
  # Fetch missing dependency libraries (lib\windows_<arch>) and Git LFS data.
  [switch]$FetchLibraries,
  # Fail instead of fetching when libraries are missing.
  [switch]$NoFetchLibraries,
  # Remove the release directory before copying the new build.
  [switch]$CleanRelease,
  # Use the Ninja generator (make.bat "ninja") instead of MSBuild.
  [switch]$Ninja,
  # Visual Studio version for make.bat, e.g. 2022. Default: newest found by make.bat.
  [string]$VisualStudio = "",
  # Skip the final "blender.exe --background --version" run.
  [switch]$NoSmokeTest,
  # Launch the built Blender afterwards.
  [switch]$Run,
  # Launch the built Blender with this .blend file afterwards.
  [string]$Open = "",
  # Extra arguments passed to make.bat as they are (for example "with_tests").
  [string[]]$MakeArgs = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3

function Log([string]$Message) { Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan }
function Die([string]$Message) { Write-Host ""; Write-Host "error: $Message" -ForegroundColor Red; exit 1 }

$RootDir = $PSScriptRoot
if (-not $RootDir) { $RootDir = (Get-Location).Path }

if ($env:OS -ne "Windows_NT") {
  Die "compile.ps1 builds on Windows. On macOS and Linux run ./compile.sh."
}

switch ($env:PROCESSOR_ARCHITECTURE) {
  "AMD64" { $Arch = "x64" }
  "ARM64" { $Arch = "arm64" }
  default { Die "Unsupported processor architecture: $($env:PROCESSOR_ARCHITECTURE)" }
}

if (-not $BuildDir) { $BuildDir = Join-Path $RootDir "build\windows_${Arch}_Release" }
if (-not $ReleaseDir) { $ReleaseDir = Join-Path $RootDir "install" }
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$ReleaseDir = [System.IO.Path]::GetFullPath($ReleaseDir)
$LibDir = Join-Path $RootDir "lib\windows_$Arch"

if ($BuildDir.Contains(" ")) {
  Die "make.bat does not support spaces in the build directory: $BuildDir"
}

function Find-Python {
  $candidates = @()
  $libPython = Join-Path $LibDir "python"
  if (Test-Path $libPython) {
    $candidates += Get-ChildItem -Path $libPython -Recurse -Filter python.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch "_d\.exe$" } | ForEach-Object { $_.FullName }
  }
  foreach ($c in $candidates) { if (Test-Path $c) { return @($c) } }
  if (Get-Command py -ErrorAction SilentlyContinue) { return @("py", "-3") }
  if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
  return $null
}

function Test-LibrariesReady {
  if (-not (Test-Path (Join-Path $LibDir ".git"))) { return $false }
  foreach ($d in @("python", "ffmpeg", "openimageio", "usd")) {
    if (-not (Test-Path (Join-Path $LibDir $d))) { return $false }
  }
  return $true
}

function Test-LfsReady {
  $startup = Join-Path $RootDir "release\datafiles\startup.blend"
  if (-not (Test-Path $startup)) { return $true }
  return ((Get-Item $startup).Length -ge 1024)
}

function Invoke-Checked([string]$File, [string[]]$Arguments) {
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) { Die "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')" }
}

# Dependency libraries and LFS data.
$fetchMode = "auto"
if ($FetchLibraries) { $fetchMode = "1" }
if ($NoFetchLibraries) { $fetchMode = "0" }

if (-not (Test-LibrariesReady)) {
  if ($fetchMode -eq "0") {
    Die "Windows $Arch dependency libraries are missing from $LibDir. Run .\compile.ps1 -FetchLibraries"
  }
  if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git is required to fetch libraries." }
  & git lfs version *> $null
  if ($LASTEXITCODE -ne 0) { Die "Git LFS is required to fetch Blender's precompiled libraries." }
  $py = Find-Python
  if (-not $py) { Die "Python 3 is required to fetch libraries (install it or the 'py' launcher)." }
  Log "Fetching official Blender Windows $Arch dependency libraries"
  Push-Location $RootDir
  try {
    $pyArgs = @()
    if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }
    Invoke-Checked $py[0] ($pyArgs + @("build_files\utils\make_update.py", "--no-blender", "--architecture=$Arch"))
  } finally { Pop-Location }
  if (-not (Test-LibrariesReady)) { Die "Dependency libraries are still incomplete after fetching: $LibDir" }
}

if (-not (Test-LfsReady)) {
  if ($fetchMode -eq "0") { Die "Blender Git LFS data is incomplete. Run: git lfs pull" }
  Log "Fetching Blender root Git LFS data"
  Push-Location $RootDir
  try { Invoke-Checked "git" @("lfs", "pull") } finally { Pop-Location }
  if (-not (Test-LfsReady)) { Die "Blender Git LFS data is still incomplete after git lfs pull." }
}

Log "Build configuration"
Write-Host "  Source:      $RootDir"
Write-Host "  Build dir:   $BuildDir"
Write-Host "  Release dir: $ReleaseDir"
Write-Host "  Libraries:   $LibDir"
Write-Host "  Generator:   $(if ($Ninja) { 'Ninja' } else { 'MSBuild' })"

# make.bat configures (first run), builds and runs the INSTALL target into <build>\bin.
$makeArguments = @("release", "builddir", $BuildDir)
if ($Ninja) { $makeArguments += "ninja" }
if ($VisualStudio) { $makeArguments += $VisualStudio }
$makeArguments += $MakeArgs

Log "Building and installing Blender (make.bat $($makeArguments -join ' '))"
Push-Location $RootDir
try {
  & cmd.exe /c "make.bat $($makeArguments -join ' ')"
  if ($LASTEXITCODE -ne 0) { Die "make.bat failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }

# Locate the installed build: <build>\bin\Release (MSBuild) or <build>\bin (Ninja).
$installed = $null
foreach ($candidate in @((Join-Path $BuildDir "bin\Release"), (Join-Path $BuildDir "bin"))) {
  if (Test-Path (Join-Path $candidate "blender.exe")) { $installed = $candidate; break }
}
if (-not $installed) { Die "blender.exe was not found under $BuildDir\bin after the build." }

if ($CleanRelease) {
  if ($ReleaseDir -eq $RootDir -or $ReleaseDir -eq $BuildDir -or $ReleaseDir.Length -le 3) {
    Die "Refusing to remove unsafe release directory: $ReleaseDir"
  }
  if (Test-Path $ReleaseDir) {
    Log "Cleaning release directory: $ReleaseDir"
    Remove-Item -Recurse -Force $ReleaseDir
  }
}

Log "Copying the build to $ReleaseDir"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
& robocopy $installed $ReleaseDir /MIR /NFL /NDL /NJH /NJS /NP /R:2 /W:1 | Out-Null
# robocopy exit codes below 8 mean success.
if ($LASTEXITCODE -ge 8) { Die "Copying the build failed (robocopy exit code $LASTEXITCODE)." }
$global:LASTEXITCODE = 0

$BlenderExe = Join-Path $ReleaseDir "blender.exe"
if (-not (Test-Path $BlenderExe)) { Die "Expected Blender executable was not created: $BlenderExe" }

Log "Verifying the installed Cycles add-on"
$addon = Get-ChildItem -Path $ReleaseDir -Recurse -Directory -Filter cycles -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "scripts\\addons_core\\cycles$" } | Select-Object -First 1
if (-not $addon) { Die "Installed Cycles add-on directory was not found in $ReleaseDir." }
foreach ($name in @("properties.py", "ui.py")) {
  $src = Join-Path $RootDir "intern\cycles\blender\addon\$name"
  $dst = Join-Path $addon.FullName $name
  if (-not (Test-Path $dst)) { Die "Installed Cycles add-on file is missing: $dst" }
  if ((Get-FileHash $src).Hash -ne (Get-FileHash $dst).Hash) { Die "Installed Cycles add-on file is stale: $name" }
}

if (-not $NoSmokeTest) {
  Log "Running smoke test"
  & $BlenderExe --background --factory-startup --version | Out-Null
  if ($LASTEXITCODE -ne 0) { Die "Smoke test failed: $BlenderExe --background --version" }
}

Log "Done"
Write-Host "Blender exe: $BlenderExe"
Write-Host "JCycles add-on from this build: .\compileCycles.ps1"

if ($Open) {
  if (-not (Test-Path $Open)) { Die "Blend file does not exist: $Open" }
  Start-Process -FilePath $BlenderExe -ArgumentList @("`"$Open`"")
} elseif ($Run) {
  Start-Process -FilePath $BlenderExe
}
