# Notes

Remaining oddities and intentional decisions after the encoder/verifier/context cleanup.

## Intentional decisions

- **"no update" verifier replies are ordinary feedback.** A "no update" reply is stored and
  conditioned on like any other feedback and the sample stays active in the rollout. Decided
  2026-07-09; do not special-case it. - this is okay
- **`forward_with_cfg` splits off 3 channels** (`model_out[:, :3]`) while latents have 4. All
  configs run `cfg_scale: 1.0`, which never calls `forward_with_cfg`, so this is currently
  unreachable from the training/eval paths. Left as-is on purpose; revisit before enabling CFG. -- this is okay
- **EMA covers only the DiT.** Encoder LoRA weights (when `freeze_encoder: false`) are saved in
  the checkpoint's `context_encoder` entry but have no EMA copy. -- this is okay
- **OmniGen DDP uses `find_unused_parameters=True`.** Full finetuning leaves some parameters
  unused on individual steps and DDP errors without it. Cost: one extra graph scan per backward. -- this is okay
- **On-policy launches need `OPENROUTER_API_KEY`** in addition to the verifier's key: the eval
  distance scorer is hardcoded to OpenRouter `google/gemini-3.1-flash-lite`
  (`verifiers/eval_metrics.make_scorer`) and is built fail-fast at trainer init on rank 0. -- this is okay

## Data regeneration required

The cleanup changed the data formats, so existing artifacts under
`/gscratch/scrubbed/sriyash/...` must be regenerated (stages 1 → 3, then update the dataset
config paths): -- will do that! also re-running everything so it should be fine! not using any old data

- Stage-2 images were center-cropped before resizing; they are now resized directly
  (CLEVR's 480×320 frames were losing their edges).
- Stage-3 context tokens moved from the `Caption: ... Feedback: ...` text format to the single
  interleaved history format (`caption, image 0, feedback 0, ...`); old token caches are stale.
  Frozen-encoder offline training auto-precomputes missing tokens at startup.
- Old on-policy rollout shards (`records.pt`) are incompatible: records now store pre-attempt
  `feedback_history`/`history_attempt_paths` without `metadata_index`, and only frozen-encoder
  runs store `context_tokens`.
- Old DiT checkpoints predate the current parameter names (`test_dit_param_names_match_pre_refactor_checkpoints`
  was re-baselined); loading them strictly will fail, which is accepted.

## Environment

- `tilelang` is declared in `pyproject.toml` but never imported anywhere in the repo.
- Never install HuggingFace `datasets` into `.venv`: the local `datasets/` package shadows it.
  OmniGen training therefore uses the separate `.venv-omni` environment, and
  `models/omni_gen.py` vendors what it needs from `OmniGen.train_helper` instead of importing it.
