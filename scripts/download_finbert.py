"""
Downloads FinBERT2 and optionally FinGPT models from HuggingFace.
Run once before starting the trading agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_finbert() -> None:
    print("==> Downloading ProsusAI/finbert (~440MB)...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        AutoTokenizer.from_pretrained("ProsusAI/finbert")
        AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("    ProsusAI/finbert downloaded successfully.")
    except ImportError:
        print("    ERROR: transformers not installed. Run: pip install transformers torch")
        sys.exit(1)


def download_fingpt_optional() -> None:
    print("\n==> FinGPT (fingpt-sentiment_llama2-13b_lora) requires ~8GB RAM.")
    answer = input("    Download FinGPT? [y/N]: ").strip().lower()
    if answer != "y":
        print("    Skipped.")
        return
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        base = "meta-llama/Llama-2-13b-hf"
        lora = "FinGPT/fingpt-sentiment_llama2-13b_lora"
        print(f"    Downloading base model {base} (requires HF token for Llama 2)...")
        AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base)
        print(f"    Applying LoRA adapter {lora}...")
        PeftModel.from_pretrained(model, lora)
        print("    FinGPT downloaded successfully.")
    except Exception as exc:
        print(f"    FinGPT download failed: {exc}")


if __name__ == "__main__":
    download_finbert()
    download_fingpt_optional()
    print("\n==> Done. FinBERT2 is ready for two-tier sentiment analysis.")
