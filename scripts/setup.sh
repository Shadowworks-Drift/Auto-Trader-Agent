#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Auto-Trader-Agent  ·  Setup Script
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment in $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip wheel setuptools

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Creating required directories"
mkdir -p data logs config

echo "==> Copying default config if not present"
if [ ! -f config/config.yaml ]; then
    cp config/default_config.yaml config/config.yaml
    echo "    Created config/config.yaml — edit this file before running!"
fi

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Install and start Ollama:"
echo "       curl -fsSL https://ollama.ai/install.sh | sh"
echo "       ollama pull deepseek-r1:7b"
echo "       ollama serve"
echo ""
echo "  2. Edit config/config.yaml to set your symbols and preferences"
echo ""
echo "  3. Run in paper trading mode:"
echo "       source .venv/bin/activate"
echo "       python main.py"
echo ""
echo "  4. Run tests:"
echo "       pytest"
