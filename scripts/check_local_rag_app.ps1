[CmdletBinding()]
param(
    [string]$LocalRagUrl = "",
    [string]$ApiKey = ""
)

$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 does not always load this assembly automatically.
Add-Type -AssemblyName System.Net.Http

if ([string]::IsNullOrWhiteSpace($LocalRagUrl)) {
    $LocalRagUrl = if ($env:LOCAL_RAG_URL) {
        $env:LOCAL_RAG_URL
    }
    else {
        "http://127.0.0.1:18080"
    }
}

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    $ApiKey = if ($env:LOCAL_RAG_API_KEY) {
        $env:LOCAL_RAG_API_KEY
    }
    else {
        "none"
    }
}

$baseUrl = $LocalRagUrl.Trim().TrimEnd("/")
$requestId = "stage10-check-$([guid]::NewGuid().ToString('N'))"
$httpClient = [System.Net.Http.HttpClient]::new()
$httpClient.Timeout = [TimeSpan]::FromSeconds(240)

function Write-Pass([string]$Message) {
    Write-Host "[PASS] $Message" -ForegroundColor Green
}

function Assert-Condition([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        throw $Message
    }
}

function Invoke-LocalRagRequest {
    param(
        [Parameter(Mandatory = $true)] [string]$Method,
        [Parameter(Mandatory = $true)] [string]$Path,
        [object]$Payload = $null,
        [string]$BearerToken = "",
        [string]$CorrelationId = ""
    )

    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::$Method,
        "$baseUrl$Path"
    )
    try {
        if (-not [string]::IsNullOrWhiteSpace($BearerToken)) {
            $request.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new(
                "Bearer",
                $BearerToken
            )
        }
        if (-not [string]::IsNullOrWhiteSpace($CorrelationId)) {
            [void]$request.Headers.TryAddWithoutValidation("X-Request-ID", $CorrelationId)
        }
        if ($null -ne $Payload) {
            $json = $Payload | ConvertTo-Json -Depth 8 -Compress
            $request.Content = [System.Net.Http.StringContent]::new(
                $json,
                [System.Text.Encoding]::UTF8,
                "application/json"
            )
        }

        $response = $httpClient.SendAsync($request).GetAwaiter().GetResult()
        $content = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return [PSCustomObject]@{
            Response = $response
            Content = $content
        }
    }
    finally {
        $request.Dispose()
    }
}

try {
    $health = Invoke-LocalRagRequest -Method "Get" -Path "/health" -CorrelationId $requestId
    Assert-Condition ($health.Response.StatusCode -eq 200) "/health expected HTTP 200"
    $healthJson = $health.Content | ConvertFrom-Json
    Assert-Condition ($healthJson.status -eq "ok") "/health returned an unexpected status"
    Assert-Condition ($healthJson.service -eq "local-rag-app") "/health returned an unexpected service"
    Assert-Condition (
        $health.Response.Headers.Contains("X-Request-ID") -and
        $health.Response.Headers.GetValues("X-Request-ID")[0] -eq $requestId
    ) "/health did not return the expected X-Request-ID"
    Write-Pass "/health returns ok and X-Request-ID"

    $models = Invoke-LocalRagRequest -Method "Get" -Path "/v1/models" -BearerToken $ApiKey
    Assert-Condition ($models.Response.StatusCode -eq 200) "/v1/models expected HTTP 200"
    $modelsJson = $models.Content | ConvertFrom-Json
    Assert-Condition ($modelsJson.data.Count -eq 1) "/v1/models returned an unexpected model count"
    Assert-Condition ($modelsJson.data[0].id -eq "local-rag") "/v1/models did not return local-rag"
    Write-Pass "/v1/models returns local-rag"

    $chatPayload = @{
        model = "local-rag"
        messages = @(
            @{
                role = "user"
                content = "Reply with exactly: local RAG application check passed. /no_think"
            }
        )
        stream = $false
        temperature = 0.2
        max_tokens = 64
    }
    $chat = Invoke-LocalRagRequest -Method "Post" -Path "/v1/chat/completions" -Payload $chatPayload -BearerToken $ApiKey
    Assert-Condition ($chat.Response.StatusCode -eq 200) "/v1/chat/completions expected HTTP 200"
    $chatJson = $chat.Content | ConvertFrom-Json
    Assert-Condition ($chatJson.object -eq "chat.completion") "/v1/chat/completions returned an invalid object type"
    Assert-Condition ($chatJson.model -eq "local-rag") "/v1/chat/completions exposed an upstream model name"
    Assert-Condition (-not [string]::IsNullOrWhiteSpace($chatJson.choices[0].message.content)) "/v1/chat/completions returned empty content"
    Write-Pass "/v1/chat/completions returns text from local-rag"

    $streamPayload = @{
        model = "local-rag"
        messages = @(
            @{
                role = "user"
                content = "List two RAG benefits. /no_think"
            }
        )
        stream = $true
        temperature = 0.2
        max_tokens = 64
    }
    $stream = Invoke-LocalRagRequest -Method "Post" -Path "/v1/chat/completions" -Payload $streamPayload -BearerToken $ApiKey
    Assert-Condition ($stream.Response.StatusCode -eq 200) "streaming chat expected HTTP 200"
    Assert-Condition ($stream.Response.Content.Headers.ContentType.MediaType -eq "text/event-stream") "streaming chat did not return text/event-stream"
    Assert-Condition ($stream.Content -match "data: ") "streaming chat did not return SSE data events"
    Assert-Condition ($stream.Content -match "data: \[DONE\]") "streaming chat did not return the [DONE] marker"
    Assert-Condition ($stream.Content -match '"model"\s*:\s*"local-rag"') "streaming chat exposed an upstream model name"
    Write-Pass "streaming chat returns SSE data and [DONE]"

    $wrongKey = "$ApiKey-invalid"
    $badAuth = Invoke-LocalRagRequest -Method "Get" -Path "/v1/models" -BearerToken $wrongKey
    Assert-Condition ($badAuth.Response.StatusCode -eq 401) "wrong API key expected HTTP 401"
    $badAuthJson = $badAuth.Content | ConvertFrom-Json
    Assert-Condition ($badAuthJson.error.code -eq "invalid_api_key") "wrong API key returned an unexpected error code"
    Write-Pass "wrong API key returns 401"

    Write-Host "All stage 10 local RAG checks passed for $baseUrl" -ForegroundColor Green
}
catch {
    Write-Error "[FAIL] $($_.Exception.Message)"
    exit 1
}
finally {
    $httpClient.Dispose()
}
