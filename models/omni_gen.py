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


def training_losses(model, x1, model_kwargs):
    """Flow-matching loss, vendored verbatim from OmniGen.train_helper.loss."""
    B = len(x1)
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
        context_dim=None,
        context_dropout_prob=None,
    ):
        from diffusers.models import AutoencoderKL
        from huggingface_hub import snapshot_download
        from OmniGen import OmniGen, OmniGenProcessor

        self.device = device
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.lora_finetune = bool(lora_finetune)
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.max_input_length_limit = int(max_input_length_limit)
        self.use_feedback_images = bool(use_feedback_images)
        self.keep_raw_resolution = bool(keep_raw_resolution)
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
            net.llm.gradient_checkpointing_enable()
        if self.lora_finetune:
            from peft import LoraConfig, get_peft_model

            from algorithms.utils import requires_grad

            requires_grad(net, False)
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
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

    def load(self, path, use_ema=False):
        from algorithms.utils import load_checkpoint

        state = load_checkpoint(path, use_ema=use_ema)
        if self.lora_finetune:
            from peft import set_peft_model_state_dict

            from algorithms.utils import unwrap_model

            set_peft_model_state_dict(unwrap_model(self.net), state)
            return [], []
        return self.net.load_state_dict(state, strict=False)

    def loss(self, batch):
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
        return training_losses(self.net, output_latents, model_kwargs)["loss"].mean()

    @torch.no_grad()
    def generate(self, batch, num_sampling_steps, cfg_scale=2.5, ddim_eta=0.0, seed=None):
        from OmniGen import OmniGenPipeline

        from algorithms.utils import unwrap_model

        pipeline = OmniGenPipeline(vae=self.vae, model=unwrap_model(self.net), processor=self.processor, device=self.device)
        images = pipeline(
            prompt=batch["caption"],
            height=self.image_height,
            width=self.image_width,
            num_inference_steps=num_sampling_steps,
            guidance_scale=cfg_scale,
            use_img_guidance=False,
            seed=seed,
        )
        return images

    def ddp(self, device):
        from torch.nn.parallel import DistributedDataParallel as DDP

        self.net = DDP(self.net, device_ids=[device], find_unused_parameters=True)

    def trainable_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

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
