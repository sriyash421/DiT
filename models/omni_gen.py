"""OmniGen behind the unified model interface. Requires the external OmniGen package (.venv-omni).

TrainDataCollator and the flow-matching training loss are vendored from OmniGen.train_helper,
which must never be imported: it pulls in HuggingFace `datasets`, which this repo's local
`datasets/` package would shadow.
"""
import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_omni_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def training_losses(model, x1, model_kwargs, x0=None):
    """Flow-matching loss, vendored verbatim from OmniGen.train_helper.loss.

    `x0` optionally supplies the noise instead of drawing it fresh. On-policy passes the SAME initial
    latent that generated the attempt in context, so the model learns the specific
    (x_T, image-from-x_T, feedback) -> corrected-image path it will actually face at inference.
    The objective stays unbiased: each stored x_T was drawn from N(0, I) independently of the target.
    """
    B = len(x1)
    if x0 is None:
        if isinstance(x1, (list, tuple)):
            x0 = [torch.randn_like(img) for img in x1]
        else:
            x0 = torch.randn_like(x1)
    u = torch.normal(mean=0.0, std=1.0, size=(B,))
    t = (1 / (1 + torch.exp(-u))).to(x1[0])

    if isinstance(x1, (list, tuple)):
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

    model_output = model(xt, t, **model_kwargs)
    if isinstance(x1, (list, tuple)):
        loss = torch.stack([((ut[i] - model_output[i]) ** 2).mean() for i in range(B)], dim=0)
    else:
        loss = mean_flat((model_output - ut) ** 2)
    return {"loss": loss}


NO_CHANGE_FEEDBACK = "no changes needed, the image already matches the prompt"

# Stand-in used when caption dropout fires on a feedback row. Deliberately a well-formed but
# contentless description: keeping the instruction's shape intact means the only thing removed is
# the information that lets the model shortcut straight to the target.
MASKED_CAPTION = "a scene"


def history_instruction(caption, feedback_history):
    """OmniGen prompt for a rollout context with interleaved history.

    No history -> the raw caption (matches text-to-image training). With N prior attempts the
    history is INTERLEAVED, repeating the prompt after every attempt:

        Prompt: <caption>. Attempted image: <|image_1|>. Original prompt: <caption>. Feedback: ...
                           Attempted image: <|image_2|>. Original prompt: <caption>. Feedback: ...

    Repeating the caption keeps it close to the end of the sequence, so attention does not lose the
    original request as the history grows (an image contributes 256 tokens, so with 3 prior attempts
    a single leading caption sits ~800 tokens back). The processor only requires the image ids to be
    1..N and continuous -- they need not be adjacent -- so interleaving is safe.

    An empty feedback (the verifier found nothing to fix) is stated explicitly rather than left
    blank, so "correct already" is a signal the model can learn rather than a missing field.
    """
    if not feedback_history:
        return caption
    parts = [f"Prompt: {caption}."]
    for idx, feedback in enumerate(feedback_history):
        text = str(feedback).strip() or NO_CHANGE_FEEDBACK
        parts.append(
            f"Attempted image: <|image_{idx + 1}|>. Original prompt: {caption}. Feedback: {text}."
        )
    return " ".join(parts)


class OmniClevrDataset(Dataset):
    """Adapts CLEVR rows to OmniGen text (and image-edit) training examples."""

    def __init__(
        self,
        dataset,
        processor,
        image_transform,
        condition_dropout_prob=0.0,
        max_input_length_limit=18000,
        use_feedback_images=True,
    ):
        self.dataset = dataset
        self.processor = processor
        self.image_transform = image_transform
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.max_input_length_limit = int(max_input_length_limit)
        self.use_feedback_images = bool(use_feedback_images)

    def __len__(self):
        return len(self.dataset)

    def _generated_image_for_index(self, idx):
        return self.dataset.generated_image_for_index(idx)

    @staticmethod
    def _feedback_instruction(item):
        caption = item["caption"].strip()
        feedback = item["feedback"].strip()
        if feedback:
            return (
                "<|image_1|> Edit the input image so it matches this CLEVR description: "
                f"{caption}. Apply this feedback: {feedback}"
            )
        return f"<|image_1|> Edit the input image so it matches this CLEVR description: {caption}."

    def _make_example(self, idx):
        item = self.dataset[idx]
        output_image = item["image"]
        input_images = None
        instruction = item["caption"]

        if self.use_feedback_images and item["is_feedback"]:
            generated = self._generated_image_for_index(idx)
            if generated is not None:
                instruction = self._feedback_instruction(item)
                input_images = [self.image_transform(generated)]

        if random.random() < self.condition_dropout_prob:
            instruction = ""
            input_images = None

        mllm_input = self.processor.process_multi_modal_prompt(instruction, input_images)
        if len(mllm_input["input_ids"]) > self.max_input_length_limit:
            raise RuntimeError(
                f"input token count {len(mllm_input['input_ids'])} exceeds "
                f"max_input_length_limit={self.max_input_length_limit}"
            )
        return {
            "mllm_input": mllm_input,
            "output_image": output_image,
            "source_index": item.get("source_index", 0),
            "is_feedback": item["is_feedback"],
        }

    def __getitem__(self, idx):
        return self._make_example(int(idx))


class OmniClevrCollator:
    """Vendored OmniGen TrainDataCollator plus source/feedback bookkeeping."""

    def __init__(self, processor, hidden_size, keep_raw_resolution=False):
        from OmniGen.processor import OmniGenCollator

        self.inner = OmniGenCollator(
            pad_token_id=processor.text_tokenizer.eos_token_id,
            hidden_size=hidden_size,
        )
        self.keep_raw_resolution = keep_raw_resolution

    def __call__(self, features):
        mllm_inputs = [item["mllm_input"] for item in features]
        output_images = [item["output_image"].unsqueeze(0) for item in features]
        target_img_size = [[x.size(-2), x.size(-1)] for x in output_images]

        (
            input_ids,
            position_ids,
            attention_mask,
            padding_images,
            pixel_values,
            image_sizes,
        ) = self.inner.process_mllm_input(mllm_inputs, target_img_size)

        if not self.keep_raw_resolution:
            output_images = torch.cat(output_images, dim=0)
            pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "input_pixel_values": pixel_values,
            "input_image_sizes": image_sizes,
            "padding_images": padding_images,
            "output_images": output_images,
            "source_index": torch.tensor([item["source_index"] for item in features], dtype=torch.long),
            "is_feedback": torch.tensor([item["is_feedback"] for item in features], dtype=torch.bool),
        }


class OmniGenModel:
    """OmniGen with its processor and VAE behind the same interface as QwenDiT."""

    def __init__(
        self,
        model_name_or_path,
        image_height,
        image_width,
        vae,
        lora_finetune,
        lora_rank,
        lora_alpha,
        lora_init,
        lora_target_modules,
        gradient_checkpointing,
        mixed_precision,
        condition_dropout_prob,
        max_input_length_limit,
        use_feedback_images,
        keep_raw_resolution,
        device,
        stored_noise_prob=1.0,
        context_dim=None,
        context_dropout_prob=None,
        base_ckpt=None,
        lora_dropout=0.0,
    ):
        from diffusers.models import AutoencoderKL
        from huggingface_hub import snapshot_download
        from OmniGen import OmniGen, OmniGenProcessor

        import OmniGen.scheduler as _sched
        _sched.tqdm = lambda iterable, *args, **kwargs: iterable  # silence OmniGen's per-DDIM-step bar

        self.device = device
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.lora_finetune = bool(lora_finetune)
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.max_input_length_limit = int(max_input_length_limit)
        self.use_feedback_images = bool(use_feedback_images)
        self.keep_raw_resolution = bool(keep_raw_resolution)
        # 1.0 = always reuse the rollout's x_T as the flow-matching noise; 0.0 = always fresh.
        self.stored_noise_prob = float(stored_noise_prob)
        self.transform = build_omni_transform(self.image_height, self.image_width)

        if os.path.exists(model_name_or_path):
            self.model_path = model_name_or_path
        else:
            self.model_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=os.getenv("HF_HUB_CACHE"),
                ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5", "model.pt"],
            )
        self.processor = OmniGenProcessor.from_pretrained(self.model_path)
        net = OmniGen.from_pretrained(self.model_path)
        net.llm.config.use_cache = False
        if gradient_checkpointing:
            # Non-reentrant is DDP-safe: reentrant checkpointing double-marks LoRA params ready under
            # DDP find_unused_parameters=True ("marked ready twice"). use_reentrant=False avoids it.
            net.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if base_ckpt:
            from algorithms.utils import load_checkpoint
            # Seed the raw OmniGen base with a prior full-finetune checkpoint BEFORE wrapping
            # with LoRA, so fresh adapters train on top of those weights. strict=True on purpose:
            # a key mismatch MUST raise, never silently skip -- a silent skip would leave the
            # base un-finetuned and train an identity-init adapter on the wrong weights.
            net.load_state_dict(load_checkpoint(base_ckpt), strict=True)
        if self.lora_finetune:
            from peft import LoraConfig, get_peft_model

            from algorithms.utils import requires_grad

            requires_grad(net, False)
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                init_lora_weights=lora_init,
                target_modules=list(lora_target_modules),
            )
            net.llm.enable_input_require_grads()
            net = get_peft_model(net, lora_config)

        self.weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision, torch.float32)
        self.net = net.to(device=device, dtype=self.weight_dtype)

        if vae is None:
            vae = os.path.join(self.model_path, "vae")
        self.vae = AutoencoderKL.from_pretrained(vae).to(device=device, dtype=torch.float32).eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self._rollout_collator = None
        self._gen_collator = None

    def load(self, path, use_ema=False):
        from algorithms.utils import load_checkpoint

        state = load_checkpoint(path, use_ema=use_ema)
        if self.lora_finetune:
            from peft import set_peft_model_state_dict

            from algorithms.utils import unwrap_model

            # Report what PEFT actually matched. Returning a hardcoded ([], []) here made train.py
            # always print "missing=0, unexpected=0", so a key-name mismatch would load NOTHING and
            # stay silent -- and because LoRA inits B=0, that failure mode is an identity adapter,
            # i.e. silently training the raw base model.
            res = set_peft_model_state_dict(unwrap_model(self.net), state)
            # PEFT delegates to load_state_dict(strict=False) on the FULL model, so `missing_keys`
            # always lists every BASE tensor too (~211 for OmniGen) -- those legitimately are not in
            # a LoRA-only checkpoint and come from the pretrained weights. Reporting them raw reads
            # like a failed load. Only a LoRA key going missing is a real problem, so filter to those;
            # `unexpected` stays raw (a non-empty list means checkpoint keys the model did not match).
            missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
            return missing, list(getattr(res, "unexpected_keys", []))
        return self.net.load_state_dict(state, strict=False)

    def loss(self, batch, weights=None):
        """Flow-matching loss. `weights` is an optional per-sample weight vector (the curriculum
        weights each chain position differently); None keeps the plain batch mean."""
        from OmniGen.utils import vae_encode, vae_encode_list

        output_images = move_to_device(batch["output_images"], self.device)
        input_pixel_values = move_to_device(batch["input_pixel_values"], self.device)
        with torch.no_grad():
            if isinstance(output_images, list):
                output_latents = vae_encode_list(self.vae, output_images, self.weight_dtype)
                input_latents = vae_encode_list(self.vae, input_pixel_values, self.weight_dtype) if input_pixel_values is not None else None
            else:
                output_latents = vae_encode(self.vae, output_images, self.weight_dtype)
                input_latents = vae_encode(self.vae, input_pixel_values, self.weight_dtype) if input_pixel_values is not None else None

        model_kwargs = move_to_device({
            "input_ids": batch["input_ids"],
            "input_img_latents": input_latents,
            "input_image_sizes": batch["input_image_sizes"],
            "attention_mask": batch["attention_mask"],
            "position_ids": batch["position_ids"],
            "padding_latent": batch["padding_images"],
            "past_key_values": None,
            "return_past_key_values": False,
        }, self.device)
        # Optional stored rollout noise (see training_losses). Falls back to fresh noise when the
        # batch has none, so offline training is unchanged.
        x0 = batch.get("init_latents")
        if x0 is not None and random.random() >= self.stored_noise_prob:
            x0 = None            # this step falls back to fresh noise (ablation / regularisation)
        if x0 is not None:
            if isinstance(output_latents, (list, tuple)):
                x0 = [n.to(device=self.device, dtype=l.dtype) if n is not None else torch.randn_like(l)
                      for n, l in zip(x0, output_latents)]
            else:
                x0 = torch.stack([n if n is not None else torch.randn(output_latents.shape[1:])
                                  for n in x0]).to(device=self.device, dtype=output_latents.dtype)
        per_sample = training_losses(self.net, output_latents, model_kwargs, x0=x0)["loss"]
        if weights is None:
            return per_sample.mean()
        # Weighted mean, normalised by the weight sum so the gradient scale does not depend on the
        # batch's position mix -- a plain (loss * w).mean() would shrink whenever a batch happened
        # to draw more low-weight early positions.
        w = torch.as_tensor(weights, device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-8)

    def anchor_loss(self, batch):
        """Hold the DRAFT step at the frozen base instead of supervising it to ground truth.

        Distils in VELOCITY space: the adapted model and the reference are evaluated at the same
        (x_t, t) and the loss is the squared difference of their predicted velocities. No sampling
        is needed, so this costs one extra forward, on draft rows only.

        The reference is the base with the LoRA adapter switched off (peft's disable_adapter()), so
        there is no second set of weights in memory. Because LoRA inits B=0 the adapter IS the
        identity at step 0, hence this loss is exactly 0 at initialisation: beta is a restoring
        force against drift, not a standing cost.

        x1 is the model's own latest attempt, never the ground truth, so no GT signal reaches the
        draft step. The reference forward goes through the UNWRAPPED module -- routing it through
        DDP would register a second forward that DDP then expects to join the backward.
        """
        from OmniGen.utils import vae_encode, vae_encode_list

        from algorithms.utils import unwrap_model

        collated = self._rollout_collate(batch, caption_dropout_prob=0.0, target="attempt")
        images = move_to_device(collated["output_images"], self.device)
        input_pixel_values = move_to_device(collated["input_pixel_values"], self.device)
        with torch.no_grad():
            if isinstance(images, list):
                x1 = vae_encode_list(self.vae, images, self.weight_dtype)
                input_latents = (vae_encode_list(self.vae, input_pixel_values, self.weight_dtype)
                                 if input_pixel_values is not None else None)
            else:
                x1 = vae_encode(self.vae, images, self.weight_dtype)
                input_latents = (vae_encode(self.vae, input_pixel_values, self.weight_dtype)
                                 if input_pixel_values is not None else None)

        model_kwargs = move_to_device({
            "input_ids": collated["input_ids"],
            "input_img_latents": input_latents,
            "input_image_sizes": collated["input_image_sizes"],
            "attention_mask": collated["attention_mask"],
            "position_ids": collated["position_ids"],
            "padding_latent": collated["padding_images"],
            "past_key_values": None,
            "return_past_key_values": False,
        }, self.device)

        # one shared noise draw and timestep, so both models are compared at the same point
        stored = collated.get("init_latents")
        if isinstance(x1, (list, tuple)):
            x0 = ([n.to(device=self.device, dtype=l.dtype) if n is not None else torch.randn_like(l)
                   for n, l in zip(stored, x1)] if stored is not None
                  else [torch.randn_like(l) for l in x1])
            B = len(x1)
            u = torch.normal(mean=0.0, std=1.0, size=(B,))
            t = (1 / (1 + torch.exp(-u))).to(x1[0])
            xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        else:
            x0 = (torch.stack([n if n is not None else torch.randn(x1.shape[1:]) for n in stored]
                              ).to(device=self.device, dtype=x1.dtype)
                  if stored is not None else torch.randn_like(x1))
            B = x1.shape[0]
            u = torch.normal(mean=0.0, std=1.0, size=(B,))
            t = (1 / (1 + torch.exp(-u))).to(x1)
            t_ = t.view(t.size(0), *([1] * (len(x1.size()) - 1)))
            xt = t_ * x1 + (1 - t_) * x0

        module = unwrap_model(self.net)
        with torch.no_grad():
            with module.disable_adapter():
                v_ref = module(xt, t, **model_kwargs)
        v_theta = self.net(xt, t, **model_kwargs)
        if isinstance(v_theta, (list, tuple)):
            return torch.stack([((a - b.detach()) ** 2).mean()
                                for a, b in zip(v_theta, v_ref)]).mean()
        return ((v_theta - v_ref.detach()) ** 2).mean()

    def rollout_loss(self, batch, weights=None, caption_dropout_prob=0.0):
        """Turn rollout rows (full history) into OmniGen edit examples and run the flow loss.

        Mirrors OmniClevrDataset/OmniClevrCollator: each row's prior attempts become input images
        for a multi-image edit instruction; the GT is the output image. Reuses self.loss.

        `caption_dropout_prob` replaces the caption with a neutral placeholder on FEEDBACK rows
        (those with a non-empty history). In clevr_g6 the caption alone fully determines the target,
        so the attempt and critique are redundant inputs and the model can reach the target without
        ever reading them -- with only 100 captions that shortcut is much cheaper than learning the
        edit operator. Hiding the caption leaves (source image + critique) as the only route, which
        is what forces the feedback pathway to carry information. Rows at position 0 are never
        dropped: they have no source image, so blanking the caption would leave nothing to condition
        on. The caption is always present at inference -- this is a train-time regulariser.
        """
        collated = self._rollout_collate(batch, caption_dropout_prob=caption_dropout_prob,
                                         target="gt")
        return self.loss(collated, weights=weights)

    def _rollout_collate(self, batch, caption_dropout_prob=0.0, target="gt"):
        """Shared feature building for rollout_loss and anchor_loss.

        `target` chooses what the flow regresses toward: "gt" is the ground-truth image (the repair
        objective); "attempt" is the model's own latest attempt, which anchor_loss uses so that
        ground truth never reaches the draft step.
        """
        if self._rollout_collator is None:
            from algorithms.utils import unwrap_model

            self._rollout_collator = OmniClevrCollator(
                self.processor,
                hidden_size=unwrap_model(self.net).llm.config.hidden_size,
                keep_raw_resolution=self.keep_raw_resolution,
            )
        features = []
        dropped = 0
        own_attempts = batch.get("own_attempt") or batch["gt_image"]
        for gt_image, own_attempt, caption, feedback_history, attempt_images in zip(
            batch["gt_image"], own_attempts, batch["caption"], batch["feedback_history"],
            batch["attempt_images"]
        ):
            if feedback_history and caption_dropout_prob > 0.0 and random.random() < caption_dropout_prob:
                caption = MASKED_CAPTION
                dropped += 1
            instruction = history_instruction(caption, feedback_history)
            input_images = [self.transform(image) for image in attempt_images] or None
            mllm_input = self.processor.process_multi_modal_prompt(instruction, input_images)
            out_img = gt_image if target == "gt" else own_attempt
            features.append({
                "mllm_input": mllm_input,
                "output_image": self.transform(out_img),
                "source_index": 0,
                "is_feedback": bool(feedback_history),
            })
        collated = self._rollout_collator(features)
        # carry the stored sampling noise through the collator (it only knows about images/text)
        if batch.get("init_latents") is not None:
            collated["init_latents"] = batch["init_latents"]
        self.last_caption_dropped = dropped
        return collated

    @torch.no_grad()
    def generate(self, batch, num_sampling_steps, cfg_scale=1.0, ddim_eta=0.0, seed=None,
                 return_latents=False, init_latents=None):
        """Single-forward (no-CFG) sampling: run the same model.forward used in training through the
        OmniGen scheduler with one conditional pass per image. Bypasses OmniGenPipeline, which always
        computes a wasted unconditional pass at cfg_scale=1.0 (2x-3x the compute). cfg_scale/ddim_eta
        are ignored (no classifier-free guidance)."""
        from OmniGen.processor import OmniGenCollator
        from OmniGen.scheduler import OmniGenScheduler
        from OmniGen.utils import vae_encode

        from algorithms.utils import unwrap_model

        module = unwrap_model(self.net)
        if self._gen_collator is None:
            self._gen_collator = OmniGenCollator(
                pad_token_id=self.processor.text_tokenizer.eos_token_id,
                hidden_size=module.llm.config.hidden_size,
            )
        captions = batch["caption"]
        feedback_histories = batch.get("feedback_history", [[] for _ in captions])
        attempt_images = batch.get("attempt_images", [[] for _ in captions])
        mllm_inputs = []
        for caption, history, images in zip(captions, feedback_histories, attempt_images):
            instruction = history_instruction(caption, history)
            input_images = [self.transform(image) for image in images] or None
            mllm_inputs.append(self.processor.process_multi_modal_prompt(instruction, input_images))

        target_size = [[self.image_height, self.image_width] for _ in captions]
        input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes = (
            self._gen_collator.process_mllm_input(mllm_inputs, target_size)
        )
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
            input_img_latents = vae_encode(self.vae, pixel_values.to(self.device), self.weight_dtype)
        else:
            input_img_latents = None

        if init_latents is not None:
            # Caller supplies x_T explicitly. Used by the on-policy rollout so every attempt in a
            # feedback chain starts from the SAME noise -- otherwise a row dropping out of `active`
            # shifts the remaining rows and silently changes their noise mid-chain.
            latents = init_latents.to(device=self.device, dtype=self.weight_dtype)
            if latents.shape[0] != len(captions):
                raise ValueError(f"init_latents has {latents.shape[0]} rows, expected {len(captions)}")
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
            latents = torch.randn(
                len(captions), 4, self.image_height // 8, self.image_width // 8,
                device=self.device, dtype=self.weight_dtype, generator=generator,
            )
        model_kwargs = move_to_device({
            "input_ids": input_ids,
            "input_img_latents": input_img_latents,
            "input_image_sizes": image_sizes,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "padding_latent": padding_images,
        }, self.device)

        scheduler = OmniGenScheduler(num_steps=int(num_sampling_steps))
        # __init__ disables the LLM KV cache for training; generation needs it (the scheduler crops
        # conditioning tokens after step 0, expecting them cached). forward_with_cfg does the same.
        # HF forces use_cache back to False whenever gradient checkpointing is enabled AND the module
        # is in train() mode (module.training), silently breaking this cache assumption and producing
        # garbage samples. Drop to eval() for the duration of sampling so the cache actually sticks.
        was_training = module.training
        module.eval()
        module.llm.config.use_cache = True
        try:
            samples = scheduler(latents, module.forward, model_kwargs, use_kv_cache=True, offload_kv_cache=False)
        finally:
            module.llm.config.use_cache = False
            module.train(was_training)

        samples = samples.to(torch.float32)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor
        samples = self.vae.decode(samples).sample
        samples = (samples * 0.5 + 0.5).clamp(0, 1)
        from PIL import Image as PILImage

        samples = (samples * 255).to("cpu", dtype=torch.uint8).permute(0, 2, 3, 1).numpy()
        images = [PILImage.fromarray(sample) for sample in samples]
        if return_latents:
            # the x_T each image was denoised from, one CPU tensor per item
            return images, [latents[i].detach().to("cpu") for i in range(len(images))]
        return images

    def ddp(self, device):
        from torch.nn.parallel import DistributedDataParallel as DDP

        self.net = DDP(self.net, device_ids=[device], find_unused_parameters=True)

    def trainable_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def lora_state_of(self, module):
        """LoRA-only state dict for a module (e.g. the EMA copy). The EMA is a deepcopy of the whole
        net, so saving its full state_dict writes all 3.8B base params -- 7.6GB per checkpoint --
        even though the base weights are frozen and identical to the pretrained model."""
        from algorithms.utils import unwrap_model

        if self.lora_finetune:
            from peft.utils import get_peft_model_state_dict

            return get_peft_model_state_dict(unwrap_model(module))
        return unwrap_model(module).state_dict()

    def checkpoint_state(self):
        from algorithms.utils import unwrap_model

        if self.lora_finetune:
            from peft.utils import get_peft_model_state_dict

            return get_peft_model_state_dict(unwrap_model(self.net))
        return unwrap_model(self.net).state_dict()

    def encoder_state(self):
        return None

    def save_extras(self, checkpoint_dir):
        from algorithms.utils import unwrap_model

        self.processor.text_tokenizer.save_pretrained(checkpoint_dir)
        unwrap_model(self.net).llm.config.save_pretrained(checkpoint_dir)

    def prepare_dataset(self, dataset):
        dataset.set_transform(self.transform)
        train_dataset = OmniClevrDataset(
            dataset,
            processor=self.processor,
            image_transform=self.transform,
            condition_dropout_prob=self.condition_dropout_prob,
            max_input_length_limit=self.max_input_length_limit,
            use_feedback_images=self.use_feedback_images,
        )
        from algorithms.utils import unwrap_model

        collate = OmniClevrCollator(
            self.processor,
            hidden_size=unwrap_model(self.net).llm.config.hidden_size,
            keep_raw_resolution=self.keep_raw_resolution,
        )
        return train_dataset, collate
