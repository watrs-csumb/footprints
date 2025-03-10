# This file must be used with "source scripts/setup.sh" *from bash*
# You cannot run it directly

# Find user's python path
if command -v python3 >/dev/null 2>&1; then
    python=python3
else
    python=python
fi

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Start virtual environment
$python -m venv $DIR/..
source $DIR/../bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
