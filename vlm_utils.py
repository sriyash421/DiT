import json
from pathlib import Path

import torch
from PIL import Image


def resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = Path(root) / path
    if rooted.exists():
        return rooted
    return path


def compact_metadata_text(metadata):
    if metadata is None:
        return None
    objects = metadata.get("objects", [])
    object_lines = []
    for obj in objects:
        object_lines.append(
            f"id {obj.get('id')}: {obj.get('size')} {obj.get('color')} "
            f"{obj.get('material')} {obj.get('shape')} ({obj.get('label')})"
        )
    orders = metadata.get("orders", {})

    def order_text(key):
        labels = []
        for idx in orders.get(key, []):
            match = next((obj for obj in objects if obj.get("id") == idx), None)
            labels.append(match.get("label", str(idx)) if match else str(idx))
        return ", ".join(labels)

    return "\n".join([
        "Objects:",
        *object_lines,
        f"Left-to-right order: {order_text('left_to_right')}",
        f"Front-to-back order: {order_text('front_to_back')}",
    ])


def load_metadata_rows(dataset_root):
    path = Path(dataset_root) / "metadata.jsonl"
    if not path.exists():
        return [], {}
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows, {row.get("image_path"): row for row in rows}


def metadata_for_row(row, metadata_rows, metadata_by_image_path):
    metadata = row.get("metadata")
    if metadata is not None:
        return metadata
    idx = row.get("metadata_index")
    if isinstance(idx, int) and 0 <= idx < len(metadata_rows):
        return metadata_rows[idx]
    return metadata_by_image_path.get(row.get("image_path") or row.get("source_image_path"))


def build_context_text(caption, metadata=None, feedback=None, include_metadata=True):
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    meta_text = compact_metadata_text(metadata) if include_metadata else None
    if meta_text:
        parts.append(f"CLEVR metadata:\n{meta_text}")
    if feedback:
        parts.append(f"Feedback: {feedback}")
    return "\n\n".join(parts)


def build_history_context_text(caption, feedback_history=None):
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    feedback_history = feedback_history or []
    if feedback_history:
        history_lines = ["Generated-image feedback history:"]
        for idx, feedback in enumerate(feedback_history):
            history_lines.append(f"Image {idx}: previous generated attempt.")
            history_lines.append(f"Feedback {idx}: {feedback}")
        parts.append("\n".join(history_lines))
    return "\n\n".join(parts)


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


def load_vlm(model_id, device, dtype="bfloat16", device_map=None):
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


@torch.no_grad()
def encode_contexts(processor, model, texts, device, images=None, max_length=None, out_dtype=torch.float16):
    messages = build_messages(texts, images)
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
