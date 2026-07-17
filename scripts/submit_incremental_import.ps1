[CmdletBinding()]
param(
    [string] $CondaEnvironment = 'incremental_rag',
    [string] $CondaExe,
    [switch] $WaitForCompletion,
    [switch] $CheckRuntime,
    [switch] $KeepWindowOpen
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Convert-UnicodeText {
    param([string] $Escaped)
    return (ConvertFrom-Json ('"{0}"' -f $Escaped))
}

function Write-UserMessage {
    param([string] $Escaped, [ConsoleColor] $Color = [ConsoleColor]::White)
    Write-Host (Convert-UnicodeText $Escaped) -ForegroundColor $Color
}

function Get-TaskProgressMessage {
    param([string] $TaskFile)
    if (-not (Test-Path -LiteralPath $TaskFile)) { return $null }
    try { $task = Get-Content -LiteralPath $TaskFile -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop } catch { return $null }
    switch ([string] $task.state) {
        'PREFLIGHT' { return (Convert-UnicodeText '\u6b63\u5728\u68c0\u67e5\u8fd0\u884c\u73af\u5883...') }
        'SNAPSHOTTING' { return (Convert-UnicodeText '\u6b63\u5728\u68c0\u67e5\u5f85\u5904\u7406\u8d44\u6599...') }
        'PROCESSING_FILES' {
            $done = if ($null -eq $task.processed_file_count) { 0 } else { $task.processed_file_count }
            $total = if ($null -eq $task.total_file_count) { 0 } else { $task.total_file_count }
            $prefix = Convert-UnicodeText '\u6b63\u5728\u5904\u7406\u8d44\u6599'
            $current = if ($task.current_file_name) { " - $($task.current_file_name)" } else { '' }
            return "$prefix ($done/$total)$current"
        }
        'BUILDING_DELTA_DB' { return (Convert-UnicodeText '\u6b63\u5728\u6784\u5efa\u672c\u6279\u5c0f\u578b\u6570\u636e\u5e93...') }
        'BUILDING_DELTA_FTS' { return (Convert-UnicodeText '\u6b63\u5728\u6784\u5efa\u672c\u6279\u68c0\u7d22\u7d22\u5f15...') }
        'BUILDING_DELTA_FAISS' { return (Convert-UnicodeText '\u6b63\u5728\u6784\u5efa\u672c\u6279\u5411\u91cf\u7d22\u5f15...') }
        'VALIDATING_DELTA' { return (Convert-UnicodeText '\u6b63\u5728\u6821\u9a8c\u672c\u6279\u8d44\u6599...') }
        'PUBLISHING' { return (Convert-UnicodeText '\u6b63\u5728\u5b89\u5168\u53d1\u5e03\u8d44\u6599...') }
        'ARCHIVING' { return (Convert-UnicodeText '\u6b63\u5728\u6821\u9a8c\u5e76\u5f52\u6863\u539f\u59cb\u8d44\u6599...') }
        'NO_CHANGES' { return (Convert-UnicodeText '\u672a\u53d1\u73b0\u9700\u8981\u91cd\u590d\u5904\u7406\u7684\u65b0\u8d44\u6599\u3002') }
        'REJECTED_SERVICE_RUNNING' { return (Convert-UnicodeText '\u672c\u5730\u68c0\u7d22\u670d\u52a1\u6b63\u5728\u8fd0\u884c\uff0c\u4e3a\u4fdd\u62a4\u8d44\u6599\u5df2\u5b89\u5168\u505c\u6b62\u3002') }
        default { return $null }
    }
}

function Show-FinalSummary {
    param([string] $FilesPath, [string] $StatusRelativePath)
    if (-not (Test-Path -LiteralPath $FilesPath)) { return }
    try { $files = (Get-Content -LiteralPath $FilesPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop).files } catch { return }
    $success = @($files | Where-Object { $_.state -eq 'ARCHIVED' }).Count
    $failed = @($files | Where-Object { $_.state -in @('FAILED', 'PUBLISHED_ARCHIVE_FAILED') }).Count
    $skipped = @($files | Where-Object { $_.state -in @('DUPLICATE_UNCHANGED', 'NOT_READY', 'UNSUPPORTED') }).Count
    Write-UserMessage '\u5904\u7406\u5b8c\u6210\u3002' Green
    Write-Host ((Convert-UnicodeText '\u5df2\u53d1\u5e03\u5e76\u5b8c\u6210\u5f52\u6863\u7684\u8d44\u6599\uff1a') + " $success") -ForegroundColor Green
    Write-Host ((Convert-UnicodeText '\u9700\u8981\u4eba\u5de5\u5173\u6ce8\u7684\u8d44\u6599\uff1a') + " $failed") -ForegroundColor $(if ($failed) { 'Yellow' } else { 'Green' })
    if ($skipped) { Write-Host ((Convert-UnicodeText '\u672a\u5904\u7406\u7684\u8d44\u6599\uff1a') + " $skipped") -ForegroundColor Yellow }
    Write-Host ((Convert-UnicodeText '\u7ed3\u679c\u8bf4\u660e\u6587\u4ef6\uff1a') + " $StatusRelativePath")
}

function Resolve-CondaExecutable {
    param([string] $RequestedPath)
    $candidates = @($RequestedPath, $env:CONDA_EXE)
    try { $candidates += (Get-Command conda -ErrorAction Stop).Source } catch { }
    $candidates += @(
        (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
        'C:\ProgramData\anaconda3\Scripts\conda.exe',
        'C:\ProgramData\miniconda3\Scripts\conda.exe'
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Conda was not found. Install the delivered Anaconda/Miniconda runtime, or set INCREMENTAL_CONDA_EXE.'
}

function Resolve-EnvironmentPython {
    param([string] $ResolvedCondaExe, [string] $EnvironmentName)
    $environmentJson = & $ResolvedCondaExe env list --json
    if ($LASTEXITCODE -ne 0) { throw 'Conda could not list its environments.' }
    $environments = ($environmentJson | ConvertFrom-Json -ErrorAction Stop).envs
    $environmentRoot = $environments | Where-Object {
        [IO.Path]::GetFileName($_).Equals($EnvironmentName, [StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if (-not $environmentRoot) {
        throw "Conda environment '$EnvironmentName' was not found."
    }
    $python = Join-Path $environmentRoot 'python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The environment '$EnvironmentName' has no python.exe."
    }
    return (Resolve-Path -LiteralPath $python).Path
}

function Test-DeliveryRuntime {
    param([string] $PythonPath)
    & $PythonPath -c "import importlib.util, sys; required=('docx','fitz','paddleocr','paddle','faiss','charset_normalizer'); missing=[name for name in required if importlib.util.find_spec(name) is None]; print(sys.executable); raise SystemExit('Missing runtime modules: '+', '.join(missing) if missing else 0)"
    if ($LASTEXITCODE -ne 0) { throw 'The incremental_rag environment is incomplete. Install the delivered offline dependency package.' }
}

try {
    Write-UserMessage '\u6b63\u5728\u68c0\u67e5\u8fd0\u884c\u73af\u5883\uff0c\u8bf7\u7a0d\u5019...' Cyan
    $requestedConda = $CondaExe
    if (-not $requestedConda) { $requestedConda = $env:INCREMENTAL_CONDA_EXE }
    $effectiveConda = Resolve-CondaExecutable $requestedConda
    $pythonPath = Resolve-EnvironmentPython $effectiveConda $CondaEnvironment
    Test-DeliveryRuntime $pythonPath
} catch {
    Write-Host "Incremental import cannot start: $($_.Exception.Message)" -ForegroundColor Red
    exit 20
}

if ($CheckRuntime) {
    Write-Host "Runtime check passed: $pythonPath"
    exit 0
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cliPath = Join-Path $projectRoot 'scripts\incremental_import.py'
try {
    $submissionJson = & $pythonPath $cliPath submit --json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $submission = $submissionJson | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Host "Task submission failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 30
}

if (-not $submission.should_start_worker) {
    Write-Host "An active task already exists: $($submission.task_id)" -ForegroundColor Yellow
    Write-Host "Status file: $($submission.status_relative_path)"
    exit 10
}

$stdoutPath = Join-Path $projectRoot $submission.worker_stdout_relative_path
$stderrPath = Join-Path $projectRoot $submission.worker_stderr_relative_path
$workerArguments = @('-u', ('"{0}"' -f $cliPath.Replace('"', '\"')), 'worker', '--task-id', $submission.task_id) -join ' '
try {
    $worker = Start-Process -FilePath $pythonPath -ArgumentList $workerArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if (-not $worker) { throw 'Start-Process did not return a Worker process.' }
} catch {
    & $pythonPath $cliPath fail-start --task-id $submission.task_id | Out-Null
    Write-Host "Worker launch failed; task lock was released: $($_.Exception.Message)" -ForegroundColor Red
    exit 30
}

Write-Host "Task submitted: $($submission.task_id)"
Write-UserMessage '\u4efb\u52a1\u5df2\u542f\u52a8\uff0c\u6b63\u5728\u5904\u7406\u8d44\u6599\u3002\u8bf7\u4e0d\u8981\u5173\u95ed\u6b64\u7a97\u53e3...' Cyan
if ($WaitForCompletion) {
    $taskFile = Join-Path $projectRoot ("rag_data\incremental\work\{0}\task.json" -f $submission.task_id)
    $lastProgress = $null
    while (-not $worker.HasExited) {
        $worker.Refresh()
        $progress = Get-TaskProgressMessage $taskFile
        if ($progress -and $progress -ne $lastProgress) {
            Write-Host $progress -ForegroundColor Cyan
            $lastProgress = $progress
        }
        Start-Sleep -Seconds 1
    }
    # HasExited can become true before PowerShell has refreshed ExitCode.  An
    # explicit wait makes the process result deterministic instead of treating
    # a null value as a customer-visible failure.
    $worker.WaitForExit()
    $worker.Refresh()
    $workerExitCode = [int] $worker.ExitCode
    if ($workerExitCode -ne 0) {
        Write-UserMessage '\u5904\u7406\u672a\u5b8c\u6210\u3002\u8bf7\u8054\u7cfb\u6280\u672f\u652f\u6301\u4eba\u5458\u3002' Red
        exit $workerExitCode
    }
    Show-FinalSummary (Join-Path $projectRoot ("rag_data\incremental\work\{0}\files.json" -f $submission.task_id)) $submission.status_relative_path
    if ($KeepWindowOpen) { Read-Host (Convert-UnicodeText '\u8bf7\u6309 Enter \u952e\u5173\u95ed\u7a97\u53e3') | Out-Null }
}
exit 0
