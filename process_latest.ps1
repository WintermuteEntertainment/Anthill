# process_latest.ps1
$downloads = "C:\Users\twwca\Downloads"
$outputDir = "X:\Anthill\Anthill\anthill-loom\datasets\processed"

# Find the latest chatgpt_conversations file
$latestFile = Get-ChildItem -Path $downloads -Filter "chatgpt_conversations_*.json" | 
               Sort-Object LastWriteTime -Descending | 
               Select-Object -First 1

if (-not $latestFile) {
    Write-Host "No chatgpt_conversations files found in Downloads!" -ForegroundColor Red
    exit 1
}

Write-Host "Processing latest file: $($latestFile.Name)" -ForegroundColor Green
Write-Host "Date: $($latestFile.LastWriteTime)" -ForegroundColor Yellow

# Create output filename with date
$dateStr = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = Join-Path $outputDir "pairs_$dateStr.jsonl"

# Run the pipeline
python prepare_datasets_parallel.py $latestFile.FullName $outputFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Output: $outputFile" -ForegroundColor Green
} else {
    Write-Host "Pipeline failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}