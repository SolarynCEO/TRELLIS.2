[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [string]$VcVarsPath = "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    [string]$TorchCudaArchList = "",
    [string]$ExtensionRoot = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$patchFile = Join-Path $repoRoot "patches/flexgemm-windows-msvc.patch"
$eigenHeader = Join-Path $repoRoot "o-voxel/third_party/eigen/Eigen/Dense"

$nvdiffrastRepo = "https://github.com/NVlabs/nvdiffrast.git"
$nvdiffrastCommit = "253ac4fcea7de5f396371124af597e6cc957bfae"  # v0.4.0
$cuMeshRepo = "https://github.com/JeffreyXiang/CuMesh.git"
$cuMeshCommit = "12289e1062f0603f2f0d0771b02e1395d247f26f"
$flexGemmRepo = "https://github.com/JeffreyXiang/FlexGEMM.git"
$flexGemmCommit = "6dd94a859c26ee8246888502eada3dd8ad85532e"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path

if (-not (Test-Path -LiteralPath $VcVarsPath)) {
    throw "vcvars64.bat not found at '$VcVarsPath'. Install VS 2022 Build Tools and/or override -VcVarsPath."
}

if (-not (Test-Path -LiteralPath $CudaHome)) {
    throw "CUDA_HOME path not found: '$CudaHome'. Install CUDA 12.4 or override -CudaHome."
}

if ([string]::IsNullOrWhiteSpace($TorchCudaArchList)) {
    if ([string]::IsNullOrWhiteSpace($env:TORCH_CUDA_ARCH_LIST)) {
        $TorchCudaArchList = "8.9"
    } else {
        $TorchCudaArchList = $env:TORCH_CUDA_ARCH_LIST
    }
}

if ([string]::IsNullOrWhiteSpace($ExtensionRoot)) {
    $ExtensionRoot = Join-Path $env:TEMP "trellis2-extensions"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $repoRoot "build-logs/windows-extensions-$timestamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Invoke-LoggedExternal {
    param(
        [Parameter(Mandatory = $true)][string]$StepName,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )

    $logFile = Join-Path $logDir "$StepName.log"
    Write-Host "[$StepName] $FilePath $($Arguments -join ' ')"
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments *>&1 | Tee-Object -FilePath $logFile
        if ($LASTEXITCODE -ne 0) {
            throw "Step '$StepName' failed with exit code $LASTEXITCODE. See: $logFile"
        }
    } finally {
        Pop-Location
    }
}

function Import-VcVarsEnvironment {
    param([Parameter(Mandatory = $true)][string]$VcVarsBatPath)

    $command = "call `"$VcVarsBatPath`" amd64 >nul && set"
    $vcEnv = & cmd.exe /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64 initialization failed (exit code $LASTEXITCODE)."
    }

    foreach ($line in $vcEnv) {
        if ($line -match "^(.*?)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

function Clone-AtCommit {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$RepoUrl,
        [Parameter(Mandatory = $true)][string]$Commit
    )

    $targetDir = Join-Path $ExtensionRoot $Name
    if (Test-Path -LiteralPath $targetDir) {
        Remove-Item -Recurse -Force $targetDir
    }

    Invoke-LoggedExternal -StepName "$Name-clone" -WorkingDirectory $ExtensionRoot -FilePath "git" -Arguments @("clone", $RepoUrl, $targetDir)
    Invoke-LoggedExternal -StepName "$Name-checkout" -WorkingDirectory $targetDir -FilePath "git" -Arguments @("checkout", $Commit)
    Invoke-LoggedExternal -StepName "$Name-submodules" -WorkingDirectory $targetDir -FilePath "git" -Arguments @("submodule", "update", "--init", "--recursive")
    return $targetDir
}

Write-Host "Initializing VS/MSVC build environment..."
Import-VcVarsEnvironment -VcVarsBatPath $VcVarsPath

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"
$env:PATH = "$CudaHome\bin;$CudaHome\libnvvp;$env:PATH"

if (Test-Path -LiteralPath $ExtensionRoot) {
    Remove-Item -Recurse -Force $ExtensionRoot
}
New-Item -ItemType Directory -Path $ExtensionRoot -Force | Out-Null

Invoke-LoggedExternal -StepName "trellis2-submodules" -WorkingDirectory $repoRoot -FilePath "git" -Arguments @("submodule", "update", "--init", "--recursive")
if (-not (Test-Path -LiteralPath $eigenHeader)) {
    throw "Eigen submodule is still missing at '$eigenHeader'. Run 'git submodule update --init --recursive' and verify your checkout."
}

Invoke-LoggedExternal -StepName "python-build-basics" -WorkingDirectory $repoRoot -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "ninja")

$nvdiffrastDir = Clone-AtCommit -Name "nvdiffrast" -RepoUrl $nvdiffrastRepo -Commit $nvdiffrastCommit
$cuMeshDir = Clone-AtCommit -Name "CuMesh" -RepoUrl $cuMeshRepo -Commit $cuMeshCommit
$flexGemmDir = Clone-AtCommit -Name "FlexGEMM" -RepoUrl $flexGemmRepo -Commit $flexGemmCommit

Invoke-LoggedExternal -StepName "nvdiffrast-build" -WorkingDirectory $repoRoot -FilePath $PythonExe -Arguments @("-m", "pip", "install", $nvdiffrastDir, "--no-build-isolation")
Invoke-LoggedExternal -StepName "cumesh-build" -WorkingDirectory $repoRoot -FilePath $PythonExe -Arguments @("-m", "pip", "install", $cuMeshDir, "--no-build-isolation")
Invoke-LoggedExternal -StepName "triton-windows-install" -WorkingDirectory $repoRoot -FilePath $PythonExe -Arguments @("-m", "pip", "install", "--upgrade", "triton-windows")
Invoke-LoggedExternal -StepName "flexgemm-patch" -WorkingDirectory $flexGemmDir -FilePath "git" -Arguments @("apply", "--whitespace=nowarn", $patchFile)
Invoke-LoggedExternal -StepName "flexgemm-build" -WorkingDirectory $repoRoot -FilePath $PythonExe -Arguments @("-m", "pip", "install", $flexGemmDir, "--no-build-isolation")
Invoke-LoggedExternal -StepName "o-voxel-build" -WorkingDirectory $repoRoot -FilePath $PythonExe -Arguments @("-m", "pip", "install", (Join-Path $repoRoot "o-voxel"), "--no-build-isolation", "--no-deps")

Write-Host ""
Write-Host "Build complete."
Write-Host "Logs are in: $logDir"
Write-Host "TORCH_CUDA_ARCH_LIST=$env:TORCH_CUDA_ARCH_LIST"
