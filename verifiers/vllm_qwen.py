"""Qwen-VL verifier served by a local vLLM OpenAI-compatible endpoint."""
from verifiers.base import OpenAIChatVerifier


class VLLMQwenVerifier(OpenAIChatVerifier):
    def __init__(
        self,
        api_url,
        model,
        api_key="EMPTY",
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
            api_key=api_key,
            model=model,
            api_url=api_url,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            workers=workers,
            enable_thinking=enable_thinking,
            include_caption=include_caption,
            include_metadata=include_metadata,
            image_size=image_size,
            extra_payload={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
        )
