# Bulk load CARE_To_Compare Wind Farm A CSVs into per-turbine tables.
# Usage example:
#   .\scripts\wfa_bulk_load.ps1 -Server "localhost\B19CL3PCQLSERVER" -Database "ACM" -DataRoot "data/CARE_To_Compare/Wind Farm A/datasets"

param(
    [string]$Server,
    [string]$Database = "ACM",
    [string]$DataRoot = "data/CARE_To_Compare/Wind Farm A/datasets"
)

if (-not (Test-Path -Path $DataRoot)) {
    Write-Error "DataRoot not found: $DataRoot";
    exit 1;
}

$csvFiles = Get-ChildItem -Path $DataRoot -Filter "*.csv" -File;

if ($csvFiles.Count -eq 0) {
    Write-Error "No CSV files found under $DataRoot";
    exit 1;
}

# Build the explicit column list (CSV header matches table schema except for time_stamp -> EntryDateTime).
$headerLine = Get-Content -Path $csvFiles[0].FullName -TotalCount 1;
$columnNames = $headerLine -split ';';
$columnNames[0] = 'EntryDateTime';
$columnList = ($columnNames | ForEach-Object { "        [{0}]" -f $_ }) -join ",`n";

foreach ($file in $csvFiles) {
    $assetId = [int]::Parse([System.IO.Path]::GetFileNameWithoutExtension($file.Name));
    $tableName = "WFA_TURBINE_${assetId}_Data";

    $query = @"
    SET QUOTED_IDENTIFIER ON;
    SET ANSI_NULLS ON;
    IF OBJECT_ID('tempdb..#WFAStage') IS NOT NULL DROP TABLE #WFAStage;
    SELECT TOP 0
$columnList
    INTO #WFAStage
    FROM [dbo].[$tableName];

    BULK INSERT #WFAStage
    FROM '$($file.FullName)'
    WITH (
        FIRSTROW = 2,
        FIELDTERMINATOR = ';',
        ROWTERMINATOR = '0x0a',
        TABLOCK
    );

    INSERT INTO [dbo].[$tableName]
    (
$columnList
    )
    SELECT
$columnList
    FROM #WFAStage;

    DROP TABLE #WFAStage;
"@;

    Write-Host "Loading $($file.Name) into $tableName ...";
    sqlcmd -S $Server -d $Database -E -Q $query;
}

Write-Host "Bulk load complete.";