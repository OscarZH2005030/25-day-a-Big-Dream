param([string]$PartsDirectory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$PartsDirectory = [System.IO.Path]::GetFullPath($PartsDirectory)
$manifestPath = Join-Path $PartsDirectory 'handoff-parts.json'
$m = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($m.archive_name -ne 'TFB_20261102.zip') { throw 'Unexpected archive name' }
if ($m.archive_sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw 'Invalid archive hash' }
$parts = @($m.parts)
if ($parts.Count -lt 1) { throw 'Empty parts manifest' }
$total = [long]0
for ($i = 0; $i -lt $parts.Count; $i++) {
    $part = $parts[$i]
    $expectedName = 'TFB_20261102.zip.part{0:D3}' -f ($i+1)
    if ($part.name -ne $expectedName) { throw 'Unsafe or unordered part filename' }
    if ($part.sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw 'Invalid part hash' }
    $p = Join-Path $PartsDirectory $part.name
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw ('Missing part: '+$part.name) }
    if ((Get-Item -LiteralPath $p).Length -ne [long]$part.bytes) { throw ('Wrong part size: '+$part.name) }
    $hash = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash
    if ($hash -ne $part.sha256) { throw ('Part checksum failed: '+$part.name) }
    $total += [long]$part.bytes
    Write-Host ('Verified {0}/{1}: {2}' -f ($i+1),$parts.Count,$part.name)
}
if ($total -ne [long]$m.archive_bytes) { throw 'Total part size mismatch' }
$archive = Join-Path $PartsDirectory 'TFB_20261102.zip'
if (Test-Path -LiteralPath $archive) {
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -eq $m.archive_sha256) {
        Write-Host ('Archive already verified: '+$archive)
        exit 0
    }
    throw 'Existing archive differs. Refusing to overwrite it.'
}
$temp = Join-Path $PartsDirectory 'TFB_20261102.zip.assembling'
if (Test-Path -LiteralPath $temp) { throw 'A partial merge already exists. Inspect it before retrying.' }
$buffer = New-Object byte[] (4MB)
$output = [System.IO.File]::Open($temp,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try {
    foreach ($part in $parts) {
        $inputStream = [System.IO.File]::OpenRead((Join-Path $PartsDirectory $part.name))
        try {
            while (($read = $inputStream.Read($buffer,0,$buffer.Length)) -gt 0) {
                $output.Write($buffer,0,$read)
            }
        } finally { $inputStream.Dispose() }
    }
} finally { $output.Dispose() }
if ((Get-Item -LiteralPath $temp).Length -ne [long]$m.archive_bytes) { throw 'Merged file length mismatch' }
if ((Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash -ne $m.archive_sha256) { throw 'Merged archive checksum failed' }
if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($temp)) -ne $PartsDirectory) { throw 'Unsafe temporary path' }
if ([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($archive)) -ne $PartsDirectory) { throw 'Unsafe archive path' }
Move-Item -LiteralPath $temp -Destination $archive
Write-Host ('Archive verified: '+$archive) -ForegroundColor Green
Write-Host 'Run Extract_Handoff.ps1 next. No training or Kaggle submission was performed.'
