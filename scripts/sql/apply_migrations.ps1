param(
    [string]$Server = "localhost",
    [string]$Database = "ACM",
    [string]$Auth = "integrated", # "integrated" or "sql"
    [string]$User = "",
    [string]$Password = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Migration {
    param(
        [string]$SqlFile
    )
    Write-Host "Applying migration: $SqlFile" -ForegroundColor Cyan
    if ($Auth -eq "sql") {
        sqlcmd -S $Server -d $Database -U $User -P $Password -b -i $SqlFile
    } else {
        sqlcmd -S $Server -d $Database -E -b -i $SqlFile
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$migrationsDir = Join-Path (Split-Path -Parent $root) "scripts/sql/migrations"

if (!(Test-Path $migrationsDir)) {
    throw "Migrations directory not found: $migrationsDir"
}

# Order by filename to ensure deterministic application (e.g., 001_*.sql, 002_*.sql)
$files = Get-ChildItem -Path $migrationsDir -Filter "*.sql" | Sort-Object Name
if ($files.Count -eq 0) {
    Write-Host "No migrations to apply." -ForegroundColor Yellow
    exit 0
}

foreach ($f in $files) {
    try {
        Invoke-Migration -SqlFile $f.FullName
    } catch {
        Write-Host "Migration failed: $($f.Name)" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

Write-Host "All migrations applied successfully." -ForegroundColor Green
