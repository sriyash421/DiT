"""Verifiers that score generated images against ground truth with a VLM."""
from verifiers.base import FeedbackVerifier, OpenAIChatVerifier, VerificationResult
from verifiers.gemini import GeminiVerifier
from verifiers.open_router import OpenRouterVerifier
from verifiers.vllm_qwen import VLLMQwenVerifier


def build_verifier(backend, **kwargs):
    """Build a verifier by backend name: gemini, vllm-qwen, open-router, or local-qwen."""
    if backend == "gemini":
        return GeminiVerifier(**kwargs)
    if backend == "vllm-qwen":
        return VLLMQwenVerifier(**kwargs)
    if backend == "open-router":
        return OpenRouterVerifier(**kwargs)
    if backend == "local-qwen":
        from verifiers.local_qwen import LocalQwenVerifier

        return LocalQwenVerifier(**kwargs)
    raise ValueError(f"Unknown verifier backend: {backend}")
