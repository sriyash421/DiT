import base64
import concurrent.futures
import io
import re
import time
from dataclasses import dataclass, field

import torch
from PIL import Image

from vlm_utils import compact_metadata_text


GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


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
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


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


def build_feedback_prompt(
    caption=None,
    metadata=None,
    include_caption=True,
    include_metadata=False,
    enable_thinking=False,
):
    caption_block = f"Caption: {caption}\n\n" if include_caption and caption else ""
    metadata_text = compact_metadata_text(metadata) if include_metadata else None
    metadata_block = f"Target CLEVR metadata:\n{metadata_text}\n\n" if metadata_text else ""
    prompt = (
        "Give feedback for a CLEVR image generator. "
        "Image 1 is correct; image 2 is generated.\n"
        f"{caption_block}"
        f"{metadata_block}"
        "Write one short command to fix image 2. "
        "Use this priority: missing/extra object > shape > color > size > material > position/depth > background. "
        "Mention one object and one edit only. Do not use and. Do not explain. "
        "Return only the command, under 12 words."
    )
    if enable_thinking:
        prompt += " If you reason, end with exactly: FINAL: <command under 12 words>."
    return prompt


def build_history_feedback_prompt(
    caption=None,
    past_feedback=None,
    include_caption=True,
    enable_thinking=False,
):
    caption_block = f"Caption: {caption}\n\n" if include_caption and caption else ""
    past_feedback = [str(item).strip() for item in (past_feedback or []) if str(item).strip()]
    if past_feedback:
        history_block = "Previous feedback already given:\n" + "\n".join(
            f"- {item}" for item in past_feedback
        ) + "\n\n"
    else:
        history_block = ""
    prompt = (
        "Give feedback for a CLEVR image generator. "
        "Image 1 is correct; image 2 is generated.\n"
        f"{caption_block}"
        f"{history_block}"
        "Write one short new command to fix image 2. "
        "Do not repeat a previous command unless that exact issue is still the clearest remaining error. "
        "Use this priority: missing/extra object > shape > color > size > material > position/depth > background. "
        "Mention one object and one edit only. Do not use and. Do not explain. "
        "Return only the command, under 12 words."
    )
    if enable_thinking:
        prompt += " If you reason, end with exactly: FINAL: <command under 12 words>."
    return prompt


def build_distance_prompt(caption=None, include_caption=True):
    caption_block = f"Caption: {caption}\n\n" if include_caption and caption else ""
    return (
        "Compare two CLEVR images. Image 1 is correct; image 2 is generated.\n"
        f"{caption_block}"
        "Estimate the minimum number of simple object edits needed to make image 2 match image 1. "
        "If image 2 already matches image 1, return 0. "
        "Count missing/extra object, shape, color, size, material, and position/depth errors as edits. "
        "Return only one integer from 0 to 9."
    )


def parse_distance_score(text):
    match = re.search(r"\b([0-9])\b", str(text))
    if not match:
        raise ValueError(f"could not parse distance score from: {text!r}")
    return float(int(match.group(1)))


def _total_tokens_from_usage(usage):
    for key in ("total_tokens", "totalTokenCount"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


class FeedbackVerifier:
    def verify_one(self, caption, metadata, gt_image, attempt_image):
        raise NotImplementedError

    def verify_batch(self, captions, metadata, gt_images, attempt_images):
        return [
            self.verify_one(caption, meta, gt, attempt)
            for caption, meta, gt, attempt in zip(captions, metadata, gt_images, attempt_images)
        ]

    def verify_history_batch(self, captions, metadata, gt_images, attempt_images, feedback_histories):
        del feedback_histories
        return self.verify_batch(captions, metadata, gt_images, attempt_images)

    def score_distance_batch(self, captions, metadata, gt_images, attempt_images):
        del captions, metadata, gt_images, attempt_images
        return []


class OpenAIChatVerifier(FeedbackVerifier):
    def __init__(
        self,
        api_key,
        model,
        api_url=OPENROUTER_API_URL,
        temperature=0.0,
        max_tokens=32,
        retries=2,
        timeout=120,
        workers=8,
        enable_thinking=False,
        include_caption=True,
        include_metadata=False,
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
        self.include_caption = bool(include_caption)
        self.include_metadata = bool(include_metadata)
        self.image_size = None if image_size is None else int(image_size)
        self.extra_payload = dict(extra_payload or {})

    def _prepare_image(self, image):
        return resize_square(image, self.image_size) if self.image_size else image

    def verify_one(self, caption, metadata, gt_image, attempt_image):
        import requests

        return self._post_prompt(
            build_feedback_prompt(
                caption=caption,
                metadata=metadata,
                include_caption=self.include_caption,
                include_metadata=self.include_metadata,
                enable_thinking=self.enable_thinking,
            ),
            gt_image,
            attempt_image,
            parse_feedback=True,
        )

    def verify_history_one(self, caption, metadata, gt_image, attempt_image, feedback_history):
        del metadata
        return self._post_prompt(
            build_history_feedback_prompt(
                caption=caption,
                past_feedback=feedback_history,
                include_caption=self.include_caption,
                enable_thinking=self.enable_thinking,
            ),
            gt_image,
            attempt_image,
            parse_feedback=True,
        )

    def score_distance_one(self, caption, metadata, gt_image, attempt_image):
        del metadata
        return self._post_prompt(
            build_distance_prompt(caption=caption, include_caption=self.include_caption),
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
                    token_count=_total_tokens_from_usage(usage),
                    token_usage=usage,
                )
            except Exception as exc:
                if attempt >= self.retries:
                    return VerificationResult(ok=False, error=repr(exc))
                time.sleep(2 ** attempt)
        return VerificationResult(ok=False, error="unreachable")

    def verify_batch(self, captions, metadata, gt_images, attempt_images):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self.verify_one, caption, meta, gt, attempt)
                for caption, meta, gt, attempt in zip(captions, metadata, gt_images, attempt_images)
            ]
            return [future.result() for future in futures]

    def verify_history_batch(self, captions, metadata, gt_images, attempt_images, feedback_histories):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self.verify_history_one, caption, meta, gt, attempt, history)
                for caption, meta, gt, attempt, history in zip(
                    captions, metadata, gt_images, attempt_images, feedback_histories
                )
            ]
            return [future.result() for future in futures]

    def score_distance_batch(self, captions, metadata, gt_images, attempt_images):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self.score_distance_one, caption, meta, gt, attempt)
                for caption, meta, gt, attempt in zip(captions, metadata, gt_images, attempt_images)
            ]
            return [future.result() for future in futures]


class GeminiNativeVerifier(FeedbackVerifier):
    def __init__(
        self,
        api_key,
        model=DEFAULT_GEMINI_MODEL,
        api_url=None,
        temperature=0.0,
        max_tokens=96,
        retries=2,
        timeout=120,
        workers=8,
        include_caption=False,
        include_metadata=True,
    ):
        if not api_key:
            raise ValueError("GeminiNativeVerifier requires an API key.")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or GEMINI_API_URL_TEMPLATE
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        self.timeout = int(timeout)
        self.workers = int(workers)
        self.include_caption = bool(include_caption)
        self.include_metadata = bool(include_metadata)

    def verify_one(self, caption, metadata, gt_image, attempt_image):
        import requests

        headers = {"Content-Type": "application/json"}
        url = self.api_url.format(model=self.model)
        params = {"key": self.api_key}
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {
                        "text": build_feedback_prompt(
                            caption=caption,
                            metadata=metadata,
                            include_caption=self.include_caption,
                            include_metadata=self.include_metadata,
                        )
                    },
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
                    token_count=_total_tokens_from_usage(usage),
                    token_usage=usage,
                )
            except Exception as exc:
                if attempt >= self.retries:
                    return VerificationResult(ok=False, error=repr(exc))
                time.sleep(2 ** attempt)
        return VerificationResult(ok=False, error="unreachable")

    def verify_batch(self, captions, metadata, gt_images, attempt_images):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self.verify_one, caption, meta, gt, attempt)
                for caption, meta, gt, attempt in zip(captions, metadata, gt_images, attempt_images)
            ]
            return [future.result() for future in futures]


class QwenLocalVerifier(FeedbackVerifier):
    def __init__(
        self,
        model_id,
        device,
        qwen_dtype="bfloat16",
        device_map="auto",
        temperature=0.0,
        max_tokens=256,
        workers=1,
        include_caption=True,
        include_metadata=False,
        enable_thinking=False,
        image_size=None,
    ):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.workers = int(workers)
        self.include_caption = bool(include_caption)
        self.include_metadata = bool(include_metadata)
        self.enable_thinking = bool(enable_thinking)
        self.image_size = None if image_size is None else int(image_size)
        errors = []
        for class_name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ):
            try:
                module = __import__("transformers", fromlist=[class_name])
                model_cls = getattr(module, class_name)
                kwargs = {"trust_remote_code": True}
                kwargs["torch_dtype"] = "auto" if qwen_dtype == "auto" else getattr(torch, qwen_dtype)
                if device_map:
                    kwargs["device_map"] = device_map
                model = model_cls.from_pretrained(model_id, **kwargs)
                if not device_map:
                    model = model.to(device)
                model.eval()
                self.model = model
                return
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        raise RuntimeError("Could not load local Qwen model. Tried:\n" + "\n".join(errors))

    def _prepare_image(self, image):
        return resize_square(image, self.image_size) if self.image_size else image.convert("RGB")

    def _messages_for(self, caption, metadata, gt_image, attempt_image):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._prepare_image(gt_image)},
                    {"type": "image", "image": self._prepare_image(attempt_image)},
                    {
                        "type": "text",
                        "text": build_feedback_prompt(
                            caption=caption,
                            metadata=metadata,
                            include_caption=self.include_caption,
                            include_metadata=self.include_metadata,
                            enable_thinking=self.enable_thinking,
                        ),
                    },
                ],
            }
        ]

    @torch.no_grad()
    def verify_batch(self, captions, metadata, gt_images, attempt_images):
        messages = [
            self._messages_for(caption, meta, gt, attempt)
            for caption, meta, gt, attempt in zip(captions, metadata, gt_images, attempt_images)
        ]
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            images = []
            for msg in messages:
                images.extend([item["image"] for item in msg[0]["content"] if item["type"] == "image"])
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
        )
        input_len = inputs["input_ids"].shape[1]
        generated = generated[:, input_len:]
        feedbacks = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [
            VerificationResult(ok=True, feedback=clean_feedback_text(feedback), token_usage={})
            for feedback in feedbacks
        ]


GeminiVerifier = GeminiNativeVerifier


def build_feedback_verifier(
    backend,
    model,
    api_url=None,
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
    device=None,
    qwen_dtype="bfloat16",
    device_map="auto",
):
    if backend == "gemini-native":
        return GeminiNativeVerifier(
            api_key=api_key,
            model=model,
            api_url=api_url,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            workers=workers,
            include_caption=include_caption,
            include_metadata=include_metadata,
        )
    if backend in {"openai-chat", "qwen-vllm"}:
        extra_payload = {}
        if backend == "qwen-vllm":
            extra_payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        return OpenAIChatVerifier(
            api_key=api_key,
            model=model,
            api_url=api_url or OPENROUTER_API_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            workers=workers,
            enable_thinking=enable_thinking,
            include_caption=include_caption,
            include_metadata=include_metadata,
            image_size=image_size,
            extra_payload=extra_payload,
        )
    if backend == "qwen-local":
        return QwenLocalVerifier(
            model_id=model,
            device=device,
            qwen_dtype=qwen_dtype,
            device_map=device_map,
            temperature=temperature,
            max_tokens=max_tokens,
            workers=workers,
            include_caption=include_caption,
            include_metadata=include_metadata,
            enable_thinking=enable_thinking,
            image_size=image_size,
        )
    raise ValueError(f"Unknown feedback verifier backend: {backend}")
