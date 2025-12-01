# update_grafana_dashboards.ps1
# Automatically updates Grafana dashboard JSON files to use _Latest views
# This fixes visualization artifacts caused by batch mode overlapping data

$dashboardDir = "c:\Users\bhadk\Documents\ACM V8 SQL\ACM\grafana_dashboards"
$dashboardFiles = Get-ChildItem -Path $dashboardDir -Filter "*.json" | Where-Object { $_.Name -notlike "*backup*" }

# Table name replacements (base table → Latest view)
$replacements = @{
    'ACM_RegimeTimeline' = 'ACM_RegimeTimeline_Latest'
    'ACM_HealthTimeline' = 'ACM_HealthTimeline_Latest'
    'ACM_Scores_Wide' = 'ACM_Scores_Wide_Latest'
    'ACM_ThresholdCrossings' = 'ACM_ThresholdCrossings_Latest'
    'ACM_DefectSummary' = 'ACM_DefectSummary_Latest'
    'ACM_Episodes' = 'ACM_Episodes_Latest'
    'ACM_SensorHotspots' = 'ACM_SensorHotspots_Latest'
}

Write-Host "=== Updating Grafana Dashboards to Use _Latest Views ===" -ForegroundColor Cyan
Write-Host ""

$totalFiles = 0
$totalReplacements = 0
$updatedFiles = @()

foreach ($file in $dashboardFiles) {
    $filePath = $file.FullName
    Write-Host "Processing: $($file.Name)" -ForegroundColor Yellow
    
    # Read file content
    $content = Get-Content -Path $filePath -Raw
    $originalContent = $content
    $fileReplacements = 0
    
    # Apply all replacements
    foreach ($tableName in $replacements.Keys) {
        $viewName = $replacements[$tableName]
        
        # Count occurrences before replacement
        $occurrences = ([regex]::Matches($content, [regex]::Escape($tableName))).Count
        
        if ($occurrences -gt 0) {
            # Replace table name with view name
            $content = $content -replace [regex]::Escape($tableName), $viewName
            
            Write-Host "  - Replaced '$tableName' with '$viewName' ($occurrences occurrences)" -ForegroundColor Green
            $fileReplacements += $occurrences
            $totalReplacements += $occurrences
        }
    }
    
    # Write back if changes were made
    if ($content -ne $originalContent) {
        # Create backup first
        $backupPath = "$filePath.backup"
        Copy-Item -Path $filePath -Destination $backupPath -Force
        Write-Host "  - Backup created: $($file.Name).backup" -ForegroundColor Gray
        
        # Write updated content
        $content | Set-Content -Path $filePath -NoNewline
        Write-Host "  - Updated file with $fileReplacements replacements" -ForegroundColor Cyan
        
        $updatedFiles += $file.Name
        $totalFiles++
    } else {
        Write-Host "  - No changes needed" -ForegroundColor DarkGray
    }
    
    Write-Host ""
}

Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Total dashboards processed: $($dashboardFiles.Count)"
Write-Host "Total dashboards updated: $totalFiles"
Write-Host "Total table name replacements: $totalReplacements"
Write-Host ""

if ($updatedFiles.Count -gt 0) {
    Write-Host "Updated files:" -ForegroundColor Green
    foreach ($fileName in $updatedFiles) {
        Write-Host "  - $fileName"
    }
    Write-Host ""
    Write-Host "Backups created with .backup extension" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Import updated dashboards to Grafana (or wait for auto-reload)"
    Write-Host "2. Verify wave patterns are gone in 'Regime Timeline' panels"
    Write-Host "3. Check health scores are stable in 'Health Index Over Time'"
    Write-Host "4. Delete .backup files once verified"
} else {
    Write-Host "No dashboards required updates" -ForegroundColor DarkGray
}
