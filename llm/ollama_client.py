"""
Ollama LLM client with:
  - Async HTTP calls via httpx
  - Automatic fallback model on failure
  - Structured JSON output parsing
  - Token usage tracking
  - Tenacity retry on transient errors
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class LLMResponse:
    content: str
    model: str
    elapsed_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parsed: Optional[Dict[str, Any]] = None  # populated if JSON was extracted


class OllamaClient:
    """Async client for Ollama's /api/chat endpoint.

    Usage::

        client = OllamaClient(base_url="http://localhost:11434", model="deepseek-r1:7b")
        resp = await client.chat([{"role": "user", "content": "Hello"}])
        print(resp.content)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:7b",
        fallback_model: str = "llama3:8b",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "OllamaClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        expect_json: bool = False,
    ) -> LLMResponse:
        """Send a chat request.  Falls back to fallback_model on failure."""
        target_model = model or self.model
        try:
            return await self._chat_request(
                messages=messages,
                model=target_model,
                system=system,
                temperature=temperature,
                expect_json=expect_json,
            )
        except Exception as exc:
            if target_model == self.fallback_model:
                raise
            logger.warning(
                f"Primary model {target_model} failed ({exc}), retrying with {self.fallback_model}"
            )
            return await self._chat_request(
                messages=messages,
                model=self.fallback_model,
                system=system,
                temperature=temperature,
                expect_json=expect_json,
            )

    async def health_check(self) -> bool:
        """Return True if Ollama server is reachable."""
        try:
            resp = await self._get_client().get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """Return list of locally available model names."""
        try:
            resp = await self._get_client().get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            return []

    # ── Internal ──────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _chat_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str],
        temperature: Optional[float],
        expect_json: bool,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if expect_json:
            payload["format"] = "json"

        t0 = time.perf_counter()
        resp = await self._get_client().post(
            f"{self.base_url}/api/chat", json=payload
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        parsed: Optional[Dict[str, Any]] = None
        if expect_json or content.strip().startswith("{"):
            parsed = _extract_json(content)

        return LLMResponse(
            content=content,
            model=model,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            parsed=parsed,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from an LLM response string."""
    # Strip <think>...</think> blocks emitted by reasoning models (e.g. deepseek-r1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to find JSON block inside markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Scan for first { ... } balanced block
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return None
