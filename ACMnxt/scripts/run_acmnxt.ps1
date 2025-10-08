Param(
  [Parameter(Mandatory=$true)][string]$Csv,
  [Parameter(Mandatory=$true)][string]$Equip,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [ValidateSet("train","score","full")][string]$Mode = "full",
  [switch]$Fast,
  [string]$ScoreCsv
)

function Invoke-OrExit($argsArray) {
  $disp = ($argsArray -join ' ')
  Write-Host "`n>> python $disp" -ForegroundColor DarkGray
  & python @argsArray
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Mode -eq "train") {
  Write-Host "[ACMnxt] Train -> Report: $Equip" -ForegroundColor Cyan
  $trainArgs = @('-m','acmnxt.cli.train','--csv', $Csv,'--equip',$Equip,'--out-dir',$OutDir)
  if ($Fast) { $trainArgs += '--fast' }
  Invoke-OrExit $trainArgs
  Invoke-OrExit @('-m','acmnxt.cli.report','--art-dir', $OutDir,'--equip',$Equip)
}
elseif ($Mode -eq "score") {
  Write-Host "[ACMnxt] Score-only -> Report: $Equip" -ForegroundColor Cyan
  $scCsv = if ($ScoreCsv) { $ScoreCsv } else { $Csv }
  Invoke-OrExit @('-m','acmnxt.cli.score','--csv', $scCsv,'--equip',$Equip,'--art-dir',$OutDir)
  Invoke-OrExit @('-m','acmnxt.cli.report','--art-dir', $OutDir,'--equip',$Equip)
}
else {
  Write-Host "[ACMnxt] Full: Train + Score + Report: $Equip" -ForegroundColor Cyan
  $trainArgs = @('-m','acmnxt.cli.train','--csv', $Csv,'--equip',$Equip,'--out-dir',$OutDir)
  if ($Fast) { $trainArgs += '--fast' }
  Invoke-OrExit $trainArgs
  $scCsv = if ($ScoreCsv) { $ScoreCsv } else { $Csv }
  Invoke-OrExit @('-m','acmnxt.cli.score','--csv', $scCsv,'--equip',$Equip,'--art-dir',$OutDir)
  Invoke-OrExit @('-m','acmnxt.cli.report','--art-dir', $OutDir,'--equip',$Equip)
}
exit 0
