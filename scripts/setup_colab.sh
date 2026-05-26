#!/usr/bin/env bash
set -euo pipefail

pip install --upgrade pip
pip install -r requirements.txt
python -c "import ee; ee.Authenticate()"
