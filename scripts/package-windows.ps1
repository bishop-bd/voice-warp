# Build a portable Windows x64 package; virtual audio drivers stay separate.
[CmdletBinding()]
param(
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') {
    throw 'Run package-windows.ps1 on Windows with the Rust MSVC toolchain and C++ Build Tools.'
}

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $projectRoot 'dist'
}
$outputRoot = [IO.Path]::GetFullPath($OutputDirectory)
$manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'Cargo.toml') -Raw
$version = [regex]::Match($manifest, '(?m)^version\s*=\s*"([0-9A-Za-z.+-]+)"').Groups[1].Value
if ([string]::IsNullOrWhiteSpace($version)) {
    throw 'Cannot read the package version from Cargo.toml.'
}

$target = 'x86_64-pc-windows-msvc'
$packageName = "voice-warp-$version-windows-x86_64"
$targetDirectory = Join-Path $projectRoot 'target'
$binaryPath = Join-Path $targetDirectory "$target\release\voice-warp.exe"
$archivePath = Join-Path $outputRoot "$packageName.zip"
$originalRustFlags = [Environment]::GetEnvironmentVariable('RUSTFLAGS', 'Process')
$originalEncodedFlags = [Environment]::GetEnvironmentVariable('CARGO_ENCODED_RUSTFLAGS', 'Process')

Push-Location -LiteralPath $projectRoot
try {
    # Explicit --target keeps proc macros on the host toolchain while the app
    # and its native dependencies use the statically linked MSVC C runtime.
    $env:RUSTFLAGS = (($originalRustFlags + ' -C target-feature=+crt-static').Trim())
    if (-not [string]::IsNullOrEmpty($originalEncodedFlags)) {
        $separator = [char]0x1f
        $env:CARGO_ENCODED_RUSTFLAGS = $originalEncodedFlags + $separator + '-C' + $separator + 'target-feature=+crt-static'
    }
    & cargo build --release --target $target --locked --target-dir $targetDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Windows release build failed with exit code $LASTEXITCODE."
    }
} finally {
    [Environment]::SetEnvironmentVariable('RUSTFLAGS', $originalRustFlags, 'Process')
    [Environment]::SetEnvironmentVariable('CARGO_ENCODED_RUSTFLAGS', $originalEncodedFlags, 'Process')
    Pop-Location
}

if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) {
    throw "The build did not produce $binaryPath."
}
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$stagingBase = Join-Path $outputRoot '.packaging'
New-Item -ItemType Directory -Path $stagingBase -Force | Out-Null
$stagingRoot = Join-Path $stagingBase ([guid]::NewGuid().ToString('N'))
$payloadDirectory = Join-Path $stagingRoot $packageName
New-Item -ItemType Directory -Path $payloadDirectory | Out-Null
try {
    Copy-Item -LiteralPath $binaryPath -Destination (Join-Path $payloadDirectory 'voice-warp.exe')
    Copy-Item -LiteralPath (Join-Path $projectRoot 'LICENSE') -Destination $payloadDirectory
    Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md') -Destination $payloadDirectory
    Copy-Item -LiteralPath (Join-Path $projectRoot 'docs') -Destination (Join-Path $payloadDirectory 'docs') -Recurse
    $binaryHash = (Get-FileHash -LiteralPath (Join-Path $payloadDirectory 'voice-warp.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText((Join-Path $payloadDirectory 'SHA256SUMS.txt'), "$binaryHash  voice-warp.exe`n", [Text.UTF8Encoding]::new($false))
    Compress-Archive -LiteralPath $payloadDirectory -DestinationPath $archivePath -Force
    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText("$archivePath.sha256", "$archiveHash  $packageName.zip`n", [Text.UTF8Encoding]::new($false))
} finally {
    # Only remove this invocation's GUID directory inside the packaging root.
    $resolvedStaging = [IO.Path]::GetFullPath($stagingRoot)
    $resolvedBase = [IO.Path]::GetFullPath($stagingBase).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedStaging.StartsWith($resolvedBase, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing to clean a staging directory outside the packaging root.'
    }
    if (Test-Path -LiteralPath $resolvedStaging) {
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}

Write-Host "Created $archivePath"
Write-Host "SHA-256: $archivePath.sha256"
Write-Host 'Virtual audio drivers are not included. See docs/WINDOWS.md in the package.'
