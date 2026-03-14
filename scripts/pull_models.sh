#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Pull recommended local LLM models via Ollama
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

check_ollama() {
    if ! command -v ollama &>/dev/null; then
        echo "Ollama not found. Install with: curl -fsSL https://ollama.ai/install.sh | sh"
        exit 1
    fi
}

check_ollama

echo "==> Pulling models (this may take a while depending on your internet connection)"
echo ""

# Tier 1: ~7-8B models (need ~8GB RAM)
echo "--> deepseek-r1:7b  (recommended primary — strong reasoning)"
ollama pull deepseek-r1:7b

echo "--> llama3:8b  (reliable fallback)"
ollama pull llama3:8b

# Tier 2: larger models — comment out if RAM is limited
# echo "--> deepseek-r1:14b  (higher accuracy, needs ~16GB RAM)"
# ollama pull deepseek-r1:14b

# echo "--> llama3:70b  (best quality, needs dedicated GPU)"
# ollama pull llama3:70b

echo ""
echo "==> Available models:"
ollama list
