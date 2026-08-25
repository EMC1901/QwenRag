[CmdletBinding()]
param([Parameter(Mandatory = $true)][string] $Runtime)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) { throw 'Frozen runtime executable is missing.' }

& $Runtime version
if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime version command failed.' }
& $Runtime diagnose-runtime
if ($LASTEXITCODE -ne 0) { throw 'Frozen runtime diagnostics command failed.' }

# Deeper parser/OCR/gateway smoke tests require the installer-provided OCR
# resource tree and fixture models, and run in the clean-VM acceptance stage.
