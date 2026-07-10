"""Gemini verifier over the native Google Generative Language REST API."""
import os
import time

from verifiers.base import (
    FeedbackVerifier,
    VerificationResult,
    build_feedback_prompt,
    clean_feedback_text,
    pil_to_png_base64,
    total_tokens_from_usage,
)

GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


class GeminiVerifier(FeedbackVerifier):
    def __init__(
        self,
        api_key_env="GEMINI_API_KEY",
        model=DEFAULT_GEMINI_MODEL,
        api_url=GEMINI_API_URL_TEMPLATE,
        temperature=0.0,
        max_tokens=96,
        retries=2,
        timeout=120,
        workers=8,
    ):
        self.api_key = os.environ[api_key_env]
        self.model = model
        self.api_url = api_url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        self.timeout = int(timeout)
        self.workers = int(workers)

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        import requests

        headers = {"Content-Type": "application/json"}
        url = self.api_url.format(model=self.model)
        params = {"key": self.api_key}
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": build_feedback_prompt(caption, feedback_history=feedback_history)},
                    {"text": "Ground-truth image:"},
                    {"inline_data": {"mime_type": "image/png", "data": pil_to_png_base64(gt_image)}},
                    {"text": "Generated image:"},
                    {"inline_data": {"mime_type": "image/png", "data": pil_to_png_base64(attempt_image)}},
                ],
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(url, headers=headers, params=params, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                parts = result["candidates"][0]["content"].get("parts", [])
                feedback = clean_feedback_text("".join(part.get("text", "") for part in parts))
                if not feedback:
                    raise ValueError("empty feedback")
                usage = result.get("usageMetadata", {}) or {}
                return VerificationResult(
                    ok=True,
                    feedback=feedback,
                    token_count=total_tokens_from_usage(usage),
                    token_usage=usage,
                )
            except Exception as exc:
                if attempt >= self.retries:
                    return VerificationResult(ok=False, error=repr(exc))
                time.sleep(2 ** attempt)
        return VerificationResult(ok=False, error="unreachable")
