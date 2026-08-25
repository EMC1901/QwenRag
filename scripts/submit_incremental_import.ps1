[CmdletBinding()]
param(
    [switch] $WaitForCompletion,
    [switch] $KeepWindowOpen
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

function Write-ProgressLine {
    param([object] $Task)
    switch ([string] $Task.state) {
        'PREFLIGHT' { '正在检查环境……'; break }
        'SNAPSHOTTING' { '正在检查待处理资料……'; break }
        'PROCESSING_FILES' {
            $done = if ($null -eq $Task.processed_file_count) { 0 } else { $Task.processed_file_count }
            $total = if ($null -eq $Task.total_file_count) { 0 } else { $Task.total_file_count }
            $name = if ($Task.current_file_name) { "：$($Task.current_file_name)" } else { '' }
            "正在处理资料 ($done/$total)$name"; break
        }
        'BUILDING_DELTA_DB' { '正在构建检索索引……'; break }
        'BUILDING_DELTA_FTS' { '正在构建检索索引……'; break }
        'BUILDING_DELTA_FAISS' { '正在构建检索索引……'; break }
        'VALIDATING_DELTA' { '正在校验资料……'; break }
        'PUBLISHING' { '正在安全发布……'; break }
        'ARCHIVING' { '正在归档原始资料……'; break }
        default { $null }
    }
}

$runtime = Join-Path (Split-Path -Parent $PSScriptRoot) 'QwenRagRuntime.exe'
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf)) {
    Write-Host '未找到 QwenRagRuntime.exe。请重新安装或联系技术支持人员。' -ForegroundColor Red
    exit 20
}

try {
    Write-Host '正在检查环境……' -ForegroundColor Cyan
    $submitted = & $runtime ingest
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $outcome = $submitted | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Host "资料入库未能启动：$($_.Exception.Message)" -ForegroundColor Red
    exit 30
}

if (-not $outcome.should_start_worker) {
    Write-Host "已有资料入库任务正在处理：$($outcome.task_id)" -ForegroundColor Yellow
    Write-Host "结果文件：$($outcome.status_relative_path)"
    exit 10
}

Write-Host "任务已启动：$($outcome.task_id)" -ForegroundColor Cyan
if ($WaitForCompletion) {
    $lastMessage = ''
    $terminalStates = @('SUCCEEDED', 'PARTIAL_SUCCESS', 'FAILED_RESUMABLE', 'NO_CHANGES', 'REJECTED_SERVICE_RUNNING')
    while (Test-Path -LiteralPath $outcome.task_file) {
        try { $task = Get-Content -LiteralPath $outcome.task_file -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop } catch { Start-Sleep -Seconds 1; continue }
        $message = Write-ProgressLine $task
        if ($message -and $message -ne $lastMessage) {
            Write-Host $message -ForegroundColor Cyan
            $lastMessage = $message
        }
        if ($terminalStates -contains [string] $task.state) { break }
        Start-Sleep -Seconds 1
    }
    $summary = if ($task.state -eq 'SUCCEEDED') { '处理完成。' } elseif ($task.state -eq 'PARTIAL_SUCCESS') { '处理完成，部分资料需要人工关注。' } else { "处理结束：$($task.state)" }
    Write-Host $summary -ForegroundColor $(if ($task.state -in @('SUCCEEDED', 'NO_CHANGES')) { 'Green' } else { 'Yellow' })
    Write-Host "结果文件：$($outcome.status_relative_path)"
}
if ($KeepWindowOpen) { Read-Host '请按 Enter 键关闭窗口' | Out-Null }
exit 0
