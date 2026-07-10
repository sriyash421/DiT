"""Verifier base class, prompt templates, parsers, and the OpenAI-compatible chat backend."""
import base64
import concurrent.futures
import io
import re
import time
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class VerificationResult:
    ok: bool
    feedback: str = ""
    score: float | None = None
    token_count: int = 0
    token_usage: dict = field(default_factory=dict)
    error: str = ""


def pil_to_png_base64(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_data_url(image):
    return "data:image/png;base64," + pil_to_png_base64(image)


def resize_square(image, size):
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def normalize_chat_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def clean_feedback_text(text):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    marker_match = re.search(r"(?:final|answer)\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if marker_match:
        text = marker_match.group(1).strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    text = text.replace("\r", "\n").strip()
    for line in text.splitlines():
        line = line.strip(" \t-*#`>\"'")
        if line:
            text = line
            break
    text = re.sub(r"\s+", " ", text).strip(" \"'")
    return text


def build_feedback_prompt(caption, feedback_history=(), enable_thinking=False):
    past_feedback = [str(item).strip() for item in (feedback_history or ()) if str(item).strip()]
    if past_feedback:
        history_block = "Previous feedback already given:\n" + "\n".join(
            f"- {item}" for item in past_feedback
        ) + "\n\n"
        command_line = (
            "Write one short new command to fix image 2. "
            "Do not repeat a previous command unless that exact issue is still the clearest remaining error. "
        )
    else:
        history_block = ""
        command_line = "Write one short command to fix image 2. "
    prompt = (
        "Give feedback for a CLEVR image generator. "
        "Image 1 is correct; image 2 is generated.\n"
        f"Caption: {caption}\n\n"
        f"{history_block}"
        f"{command_line}"
        "Use this priority: missing/extra object > shape > color > size > material > position/depth > background. "
        "Mention one object and one edit only. Do not use and. Do not explain. "
        "If image 2 already follows the caption and is similar to image 1, return exactly: no update. "
        "Return only the command, under 12 words."
    )
    if enable_thinking:
        prompt += " If you reason, end with exactly: FINAL: <command under 12 words>."
    return prompt


def build_distance_prompt(caption):
    return (
        "Compare two CLEVR images. Image 1 is correct; image 2 is generated.\n"
        f"Caption: {caption}\n\n"
        "Estimate the minimum number of simple object edits needed to make image 2 match image 1. "
        "If image 2 already matches image 1, return 0. "
        "Count missing/extra object, shape, color, size, material, and position/depth errors as edits. "
        "Return only one integer from 0 to 9."
    )


def parse_distance_score(text):
    text = str(text).strip()
    if re.fullmatch(r"[0-9]", text):
        return float(int(text))
    matches = re.findall(r"\b([0-9])\b", text)
    if not matches:
        raise ValueError(f"could not parse distance score from: {text!r}")
    return float(int(matches[-1]))


def total_tokens_from_usage(usage):
    for key in ("total_tokens", "totalTokenCount"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


class FeedbackVerifier:
    """Base verifier: subclasses implement the per-request _verify_row/_distance_row hooks
    (or override verify wholesale for natively batched backends); batches are threaded."""

    workers = 1

    def _map(self, fn, *columns):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(fn, *args) for args in zip(*columns)]
            return [future.result() for future in futures]

    def verify(self, captions, gt_images, attempt_images, feedback_histories=None):
        if feedback_histories is None:
            feedback_histories = [[] for _ in captions]
        assert len(captions) == len(gt_images) == len(attempt_images) == len(feedback_histories)
        return self._map(self._verify_row, captions, gt_images, attempt_images, feedback_histories)

    def score_distance(self, captions, gt_images, attempt_images):
        assert len(captions) == len(gt_images) == len(attempt_images)
        return self._map(self._distance_row, captions, gt_images, attempt_images)

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        raise NotImplementedError

    def _distance_row(self, caption, gt_image, attempt_image):
        raise NotImplementedError


class OpenAIChatVerifier(FeedbackVerifier):
    """Verifier over an OpenAI-compatible /chat/completions endpoint (OpenRouter, vLLM, ...)."""

    def __init__(
        self,
        api_key,
        model,
        api_url,
        temperature=0.0,
        max_tokens=32,
        retries=2,
        timeout=120,
        workers=8,
        enable_thinking=False,
        image_size=None,
        extra_payload=None,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = normalize_chat_url(api_url)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        self.timeout = int(timeout)
        self.workers = int(workers)
        self.enable_thinking = bool(enable_thinking)
        self.image_size = None if image_size is None else int(image_size)
        self.extra_payload = dict(extra_payload or {})

    def _prepare_image(self, image):
        return resize_square(image, self.image_size) if self.image_size else image

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        return self._post_prompt(
            build_feedback_prompt(
                caption,
                feedback_history=feedback_history,
                enable_thinking=self.enable_thinking,
            ),
            gt_image,
            attempt_image,
            parse_feedback=True,
        )

    def _distance_row(self, caption, gt_image, attempt_image):
        return self._post_prompt(
            build_distance_prompt(caption),
            gt_image,
            attempt_image,
            parse_feedback=False,
        )

    def _post_prompt(self, prompt, gt_image, attempt_image, parse_feedback=True):
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Ground-truth image:"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(self._prepare_image(gt_image))}},
            {"type": "text", "text": "Generated image:"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(self._prepare_image(attempt_image))}},
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_payload)
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                usage = result.get("usage", {}) or {}
                text = result["choices"][0]["message"]["content"]
                feedback = clean_feedback_text(text) if parse_feedback else str(text).strip()
                if not feedback and parse_feedback:
                    raise ValueError("empty feedback")
                score = None
                if not parse_feedback:
                    score = parse_distance_score(feedback)
                return VerificationResult(
                    ok=True,
                    feedback=feedback,
                    score=score,
                    token_count=total_tokens_from_usage(usage),
                    token_usage=usage,
                )
            except Exception as exc:
                if attempt >= self.retries:
                    return VerificationResult(ok=False, error=repr(exc))
                time.sleep(2 ** attempt)
        return VerificationResult(ok=False, error="unreachable")
