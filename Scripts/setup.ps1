# Run as ./scripts/setup.ps1 in PowerShell

# Note down script path
$scriptPath = $MyInvocation.MyCommand.Path

# Find user's python path
$pythonPath = (Get-Command python).Source

# Create virtual environment
& $pythonPath -m venv $scriptPath/..
# Activate virtual environment
& scriptPath\Activate.ps1
# Upgrade pip
& $pythonPath -m pip install --upgrade pip
# Install dependencies
& $pythonPath -m pip install -r requirements.txt
