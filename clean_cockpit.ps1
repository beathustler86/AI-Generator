# 🚀 Activate virtual environment
$venvPath = "F:\SoftwareDevelopment\AI Models Image\AIGenerator\.venv\Scripts\Activate.ps1"
Invoke-Expression "& '$venvPath'"

Write-Host "`n🧹 Starting cockpit cleanup..."

# 🔥 Remove pip cache
Write-Host "🧼 Clearing pip cache..."
pip cache purge

# 🧼 Remove Python bytecode
Write-Host "🧼 Removing orphaned .pyc files..."
Get-ChildItem -Recurse -Include *.pyc | Remove-Item -Force

# 🧾 Remove temp/log files
Write-Host "🧼 Removing .log and .tmp files..."
Get-ChildItem -Recurse -Include *.log, *.tmp | Remove-Item -Force

# 🧪 List untracked files (not in .gitignore or manifest)
Write-Host "`n🔍 Listing untracked files..."
git status --porcelain | Where-Object { $_ -match '^\?\?' }

# 🧼 Remove __pycache__ folders
Write-Host "🧼 Removing __pycache__ folders..."
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# 🧹 Optional: uninstall unused packages (manual review)
Write-Host "`n📦 Listing installed packages for review..."
pip list --not-required

Write-Host "`n✅ Cockpit cleanup complete."