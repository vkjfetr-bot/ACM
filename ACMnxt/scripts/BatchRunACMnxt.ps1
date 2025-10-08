Param(
  [Parameter(Mandatory=$true)][string]$CsvFolder,
  [Parameter(Mandatory=$true)][string]$OutRoot
)
$files = Get-ChildItem -Path $CsvFolder -Filter *.csv
foreach ($f in $files) {
  $equip = ($f.BaseName)
  $outDir = Join-Path $OutRoot $equip
  ./acmnxt/scripts/run_acmnxt.ps1 -Csv $f.FullName -Equip $equip -OutDir $outDir
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit 0

