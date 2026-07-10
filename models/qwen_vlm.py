"""Qwen VLM loading and context-token encoding (frozen or LoRA-finetuned)."""
from contextlib import nullcontext

import torch


def load_vlm(model_id, device, dtype="bfloat16", device_map=None):
    """Load a frozen Qwen VLM and its processor, trying the transformers auto classes in order."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    errors = []
    dtype_arg = "auto" if dtype == "auto" else getattr(torch, dtype)
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ):
        try:
            module = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(module, class_name)
            kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype_arg,
            }
            if device_map:
                kwargs["device_map"] = device_map
            model = model_cls.from_pretrained(model_id, **kwargs)
            if not device_map:
                model = model.to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            return processor, model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Could not load VLM model. Tried:\n" + "\n".join(errors))


def build_messages(texts, images=None):
    images = images or [None] * len(texts)
    messages = []
    for text, image in zip(texts, images):
        content = []
        if isinstance(image, (list, tuple)):
            for item in image:
                if item is not None:
                    content.append({"type": "image", "image": item.convert("RGB")})
        elif image is not None:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": text})
        messages.append([{"role": "user", "content": content}])
    return messages


def build_history_messages(captions, feedback_histories, image_histories):
    """One message per row in causal order: caption, then attempt image 0, feedback 0, image 1, feedback 1, ..."""
    messages = []
    for caption, feedbacks, images in zip(captions, feedback_histories, image_histories):
        content = [{"type": "text", "text": f"Caption: {caption}"}]
        for idx, (image, feedback) in enumerate(zip(images, feedbacks)):
            content.append({"type": "image", "image": image.convert("RGB")})
            content.append({"type": "text", "text": f"Feedback {idx}: {feedback}"})
        messages.append([{"role": "user", "content": content}])
    return messages


def processor_inputs(processor, messages):
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        for message in messages
    ]
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        images = []
        for message in messages:
            row_images = [item["image"] for item in message[0]["content"] if item["type"] == "image"]
            if row_images:
                images.extend(row_images)
        kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        return processor(**kwargs)


class QwenEncoder:
    """Encodes caption texts into context tokens. Frozen by default; LoRA-finetuned when freeze=False."""

    def __init__(
        self,
        model_id,
        device,
        dtype="bfloat16",
        max_length=4096,
        freeze=True,
        lora_rank=16,
        lora_alpha=16,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    ):
        self.freeze = bool(freeze)
        self.max_length = int(max_length)
        self.processor, self.model = load_vlm(model_id, device, dtype=dtype)
        if not self.freeze:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                target_modules=list(lora_target_modules),
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.train()
        base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        self.hidden_size = base.config.text_config.hidden_size

    def encode(self, texts, images):
        """Encode texts with their attempt images into (tokens, mask); gradients flow when not frozen.

        images has one entry per text: the generated attempt image(s) for feedback rows,
        None for caption-only rows.
        """
        return self._encode_messages(build_messages(texts, images))

    def encode_history(self, captions, feedback_histories, attempt_image_histories):
        """Encode interleaved history rows: caption, then each attempt image followed by its feedback."""
        return self._encode_messages(build_history_messages(captions, feedback_histories, attempt_image_histories))

    def _encode_messages(self, messages):
        module = self.model.module if hasattr(self.model, "module") else self.model
        inputs = processor_inputs(self.processor, messages)
        if inputs["input_ids"].shape[1] > self.max_length:
            for key in ("input_ids", "attention_mask"):
                inputs[key] = inputs[key][:, :self.max_length]
        model_device = next(module.parameters()).device
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad() if self.freeze else nullcontext():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].bool()
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, mask


@torch.no_grad()
def encode_contexts(processor, model, texts, device, images=None, max_length=None, out_dtype=torch.float16):
    """Encode texts (and optional images) with the frozen VLM; returns (hidden states, attention mask) on CPU."""
    return encode_messages(processor, model, build_messages(texts, images), max_length=max_length, out_dtype=out_dtype)


@torch.no_grad()
def encode_messages(processor, model, messages, max_length=None, out_dtype=torch.float16):
    """Encode prebuilt chat messages with the frozen VLM; returns (hidden states, attention mask) on CPU."""
    inputs = processor_inputs(processor, messages)
    if max_length is not None and "input_ids" in inputs and inputs["input_ids"].shape[1] > max_length:
        for key in ("input_ids", "attention_mask"):
            if key in inputs:
                inputs[key] = inputs[key][:, :max_length]
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
    else:
        mask = mask.bool()
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    return hidden.detach().cpu().to(out_dtype), mask.detach().cpu()
