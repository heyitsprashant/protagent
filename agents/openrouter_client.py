"""
OpenRouter chat-completions client shared by LLM-backed agents.
"""

from __future__ import annotations

import logging
import os

import requests


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODELS = (
    "deepseek/deepseek-v4-flash:free,"
    "meta-llama/llama-3.3-70b-instruct:free,"
    "meta-llama/llama-3.2-3b-instruct:free,"
    "openai/gpt-oss-20b:free"
)


def get_openrouter_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not found in environment.\n"
            "Get a key at: https://openrouter.ai/keys\n"
            "Add to .env: OPENROUTER_API_KEY=your_key_here"
        )
    return api_key


def get_openrouter_models() -> list[str]:
    configured = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODELS)
    return [model.strip() for model in configured.split(",") if model.strip()]


def call_openrouter(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    logger: logging.Logger,
    temperature: float,
    max_tokens: int,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ProtAgent",
    }

    for model in get_openrouter_models():
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code in {404, 408, 409, 429, 500, 502, 503, 504}:
                logger.warning(
                    "OpenRouter model %s returned retryable status %s",
                    model,
                    resp.status_code,
                )
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError:
            logger.error(
                "OpenRouter HTTP error for model %s: status=%s - %s",
                model,
                resp.status_code,
                resp.text[:300],
            )
            return None
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Unexpected OpenRouter response structure: %s", e)
            return None
        except requests.RequestException as e:
            logger.warning("OpenRouter request failed for model %s: %s", model, e)

    logger.error("OpenRouter request failed for all configured models.")
    return None
