"""Verifier over the OpenRouter chat completions API."""
import os

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
        include_caption=True,
        include_metadata=False,
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
            include_caption=include_caption,
            include_metadata=include_metadata,
            image_size=image_size,
        )
