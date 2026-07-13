"""Verifier over the OpenRouter chat completions API."""
import os
import threading

from verifiers.base import OpenAIChatVerifier

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterVerifier(OpenAIChatVerifier):
    def __init__(
        self,
        model,
        api_key_env="OPENROUTER_API_KEY",
        temperature=0.0,
        max_tokens=32,
        retries=2,
        timeout=120,
        workers=8,
        enable_thinking=False,
        image_size=None,
    ):
        super().__init__(
            api_key=os.environ[api_key_env],
            model=model,
            api_url=OPENROUTER_API_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            workers=workers,
            enable_thinking=enable_thinking,
            image_size=image_size,
            # Ask OpenRouter to return the USD cost of each call in usage.cost.
            extra_payload={"usage": {"include": True}},
        )
        # Running USD cost of all calls this verifier has made this session (requests run in a
        # thread pool, so guard the accumulator). The trainer reads and logs it.
        self.session_cost = 0.0
        self._cost_lock = threading.Lock()

    def _record_usage(self, usage):
        cost = usage.get("cost")
        if cost:
            with self._cost_lock:
                self.session_cost += float(cost)
