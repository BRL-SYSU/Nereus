"""Minimal APIYi OpenAI-compatible image client."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import requests


class APIYiRequestError(RuntimeError):
    """A terminal API error after retry handling."""


@dataclass(frozen=True)
class APIYiResponse:
    content: str
    response_id: str
    response_model: str
    system_fingerprint: str
    usage: dict[str, Any]
    latency_seconds: float
    request_attempts: int


def build_payload(
    *,
    model: str,
    prompt: str,
    image_data_uri: str,
    task: str,
) -> dict[str, Any]:
    if task not in {"count", "grid"}:
        raise ValueError(f"unsupported task: {task}")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri},
                    },
                ],
            }
        ],
    }
    if model.lower().startswith("gpt-5"):
        payload["max_completion_tokens"] = 1024
    else:
        payload["max_tokens"] = 1024
    return payload


class APIYiClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.apiyi.com/v1",
        timeout_seconds: float = 180.0,
        max_retries: int = 5,
    ) -> None:
        if not api_key.strip():
            raise ValueError("empty API key")
        self._api_key = api_key.strip()
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)

    def complete_image(
        self,
        *,
        model: str,
        prompt: str,
        image_data_uri: str,
        task: str,
    ) -> APIYiResponse:
        payload = build_payload(
            model=model,
            prompt=prompt,
            image_data_uri=image_data_uri,
            task=task,
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = time.monotonic()
        last_error = ""
        total_attempts = self._max_retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                response = requests.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException as error:
                last_error = f"{type(error).__name__}: {error}"
                retryable = True
            else:
                retryable = (
                    response.status_code == 429
                    or 500 <= response.status_code < 600
                )
                if response.status_code == 200:
                    try:
                        body = response.json()
                        content = body["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError) as error:
                        raise APIYiRequestError(
                            "HTTP 200 response did not match Chat Completions "
                            f"schema: {error}"
                        ) from error
                    return APIYiResponse(
                        content=str(content),
                        response_id=str(body.get("id", "")),
                        response_model=str(body.get("model", "")),
                        system_fingerprint=str(
                            body.get("system_fingerprint", "") or ""
                        ),
                        usage=(
                            dict(body["usage"])
                            if isinstance(body.get("usage"), dict)
                            else {}
                        ),
                        latency_seconds=time.monotonic() - started,
                        request_attempts=attempt,
                    )
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = response.text[:2000]
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{str(error_body)[:2000]}"
                )
            if not retryable or attempt == total_attempts:
                break
            delay = min(30.0, 2.0 ** (attempt - 1))
            delay += random.uniform(0.0, 0.25 * delay)
            time.sleep(delay)
        raise APIYiRequestError(
            f"request failed after {attempt} attempt(s): {last_error}"
        )

