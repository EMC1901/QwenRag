[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    # This verifies that the active shell is using rag_env (or another environment
    # with the declared runtime dependencies) before it starts a long-running server.
    & python -c "import fastapi, uvicorn, pydantic_settings"
    if ($LASTEXITCODE -ne 0) {
        throw "The active Python environment is missing local RAG dependencies. Activate rag_env before starting the application."
    }

    $settingsJson = & python -c "import json; from local_rag_app.config import get_settings; settings = get_settings(); print(json.dumps({'host': settings.local_rag_host, 'port': settings.local_rag_port}))"
    if ($LASTEXITCODE -ne 0) {
        throw "Configuration validation failed. Correct the deployed QwenRAG configuration and rerun this script."
    }

    $settings = $settingsJson | ConvertFrom-Json
    if ($settings.host -notin @("127.0.0.1", "::1")) {
        throw "Refusing to start on non-loopback host '$($settings.host)'."
    }

    Write-Host "Starting local RAG app on http://$($settings.host):$($settings.port)"
    & python -m uvicorn local_rag_app.main:app --host $settings.host --port $settings.port
}
finally {
    Pop-Location
}
