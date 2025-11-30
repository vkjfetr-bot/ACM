# Monitor ACM Batch Jobs
# Run this to check the status of batch processing jobs

Write-Host "`n=== ACM Batch Processing Monitor ===" -ForegroundColor Cyan
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Gray

$jobs = Get-Job | Where-Object { $_.Name -like "ACM_*" }

if ($jobs.Count -eq 0) {
    Write-Host "No ACM batch jobs running" -ForegroundColor Yellow
    exit
}

foreach ($job in $jobs) {
    Write-Host "`n--- $($job.Name) ---" -ForegroundColor Cyan
    Write-Host "  Status: $($job.State)" -ForegroundColor $(if ($job.State -eq "Running") { "Green" } elseif ($job.State -eq "Completed") { "Blue" } else { "Red" })
    Write-Host "  Job ID: $($job.Id)"
    Write-Host "  Started: $($job.PSBeginTime)"
    
    if ($job.HasMoreData) {
        Write-Host "`n  Recent Output:" -ForegroundColor Yellow
        $output = Receive-Job -Id $job.Id -Keep | Select-Object -Last 10
        $output | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    }
}

Write-Host "`n=== Commands ===" -ForegroundColor Cyan
Write-Host "  View full output:  Receive-Job -Name ACM_GAS_TURBINE -Keep"
Write-Host "  View full output:  Receive-Job -Name ACM_FD_FAN -Keep"
Write-Host "  Stop a job:        Stop-Job -Name ACM_GAS_TURBINE"
Write-Host "  Remove jobs:       Get-Job | Remove-Job -Force"
Write-Host "  Monitor again:     .\scripts\monitor_batch_jobs.ps1`n"
