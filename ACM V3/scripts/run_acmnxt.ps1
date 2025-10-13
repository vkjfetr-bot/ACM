<#
.SYNOPSIS
    Runs the ACMnxt (V3) pipeline: train, score, and optionally report.
.DESCRIPTION
    This wrapper script orchestrates the core ACMnxt commands, providing a single
    entry point for processing equipment data. It ensures that the core Python
    scripts are called in the correct sequence with the necessary arguments.
.PARAMETER Equip
    The identifier for the equipment being processed.
.PARAMETER TrainCsv
    The path to the CSV file containing training data.
.PARAMETER ScoreCsv
    The path to the CSV file containing scoring data.
.PARAMETER ArtDir
    The root directory where artifacts for this equipment will be stored.
.PARAMETER EnableReport
    A switch to enable the generation of the basic HTML report after scoring.
.EXAMPLE
    .\run_acmnxt.ps1 -Equip E123 -TrainCsv .\data\E123_train.csv -ScoreCsv .\data\E123_score.csv -ArtDir .\artifacts\E123 -EnableReport
    This command will train a model for E123, score new data, and generate a report, storing all outputs in .\artifacts\E123.
 #>
param(
    [Parameter(Mandatory=$true)]
    [string]$Equip,

    [Parameter(Mandatory=$true)]
    [string]$TrainCsv,

    [Parameter(Mandatory=$true)]
    [string]$ScoreCsv,

    [Parameter(Mandatory=$true)]
    [string]$ArtDir,

    [switch]$EnableReport
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CoreScriptPath = Join-Path -Path (Resolve-Path (Join-Path $ScriptRoot "..")) -ChildPath "acm_core_local_2.py"
$ReportScriptPath = Join-Path -Path (Resolve-Path (Join-Path $ScriptRoot "..")) -ChildPath "acm_report_basic.py"

Write-Host "--- Starting ACMnxt V3 Pipeline for $Equip ---"

Write-Host "[1/3] Training model..."
python $CoreScriptPath train --csv $TrainCsv --equip $Equip --prefix $Equip --art-dir $ArtDir

Write-Host "[2/3] Scoring data..."
python $CoreScriptPath score --csv $ScoreCsv --equip $Equip --prefix $Equip --manifest-prefix $Equip --art-dir $ArtDir

if ($EnableReport) {
    Write-Host "[3/3] Generating report..."
    python $ReportScriptPath --artifacts $ArtDir --equip $Equip
}

Write-Host "--- ACMnxt V3 Pipeline for $Equip finished. ---"