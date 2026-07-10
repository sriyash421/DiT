"""Qwen-VL verifier running locally through transformers."""
import torch

from models.qwen_vlm import load_vlm
from verifiers.base import FeedbackVerifier, VerificationResult, build_feedback_prompt, clean_feedback_text, resize_square


class LocalQwenVerifier(FeedbackVerifier):
    def __init__(
        self,
        model_id,
        device,
        qwen_dtype="bfloat16",
        device_map="auto",
        temperature=0.0,
        max_tokens=256,
        workers=1,
        enable_thinking=False,
        image_size=None,
    ):
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.workers = int(workers)
        self.enable_thinking = bool(enable_thinking)
        self.image_size = None if image_size is None else int(image_size)
        self.processor, self.model = load_vlm(model_id, device, dtype=qwen_dtype, device_map=device_map)

    def _prepare_image(self, image):
        return resize_square(image, self.image_size) if self.image_size else image.convert("RGB")

    def _messages_for(self, caption, gt_image, attempt_image, feedback_history):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._prepare_image(gt_image)},
                    {"type": "image", "image": self._prepare_image(attempt_image)},
                    {
                        "type": "text",
                        "text": build_feedback_prompt(
                            caption,
                            feedback_history=feedback_history,
                            enable_thinking=self.enable_thinking,
                        ),
                    },
                ],
            }
        ]

    @torch.no_grad()
    def verify(self, captions, gt_images, attempt_images, feedback_histories=None):
        if feedback_histories is None:
            feedback_histories = [[] for _ in captions]
        assert len(captions) == len(gt_images) == len(attempt_images) == len(feedback_histories)
        messages = [
            self._messages_for(caption, gt, attempt, history)
            for caption, gt, attempt, history in zip(captions, gt_images, attempt_images, feedback_histories)
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
