# CLEVR dataset pipeline

Three stages turn raw CLEVR into a training-ready `data.zarr`. All commands run from the repo root.

Download raw CLEVR first:

```bash
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip && unzip CLEVR_v1.0.zip
```

## Stage 1 — filter and caption

Keeps every scene with at most `--max-objects` objects, symlinks the images, and renders the single
**chain** caption per image (`objects: ... horizontal: ... depth: ...`; the `horizontal`/`depth`
sections are dropped for single-object scenes). Writes `images/` and `metadata.jsonl`.

```bash
python -m datasets.clevr.preprocess_clevr \
    --clevr-root CLEVR_v1.0 \
    --out /path/to/clevr_dit_dataset \
    --max-objects 3 --splits train val
```

## Stage 2 — convert to zarr

Writes images, captions, and row metadata into `data.zarr`. No GPU needed.

```bash
python -m datasets.clevr.convert_to_zarr --dataset /path/to/clevr_dit_dataset --overwrite
```

Feedback datasets (rows produced by `helper_scripts/generate_feedback_dataset.py` as `feedback_dataset.jsonl`)
use `--mode feedback`, which additionally stores the generated attempt images.

## Stage 3 — encode context tokens

Encodes each row with a frozen Qwen VLM in the single interleaved history format — base rows are
caption-only, feedback rows are `caption, generated image, feedback` — and appends the token arrays
to the same zarr. Needs a GPU. Only frozen-encoder offline training consumes these tokens, and
`train.py` runs this stage automatically at startup when the zarr lacks them; the CLI just lets you
pay the cost up front.

```bash
python -m datasets.clevr.encode_context --zarr /path/to/clevr_dit_dataset/data.zarr --batch-size 4
```

## Zarr schema (format_version 2)

```
attrs: format_version, dataset_type (base|feedback), image_size, rows, split_names
       vlm_model, context_dim, context_dtype, max_context_len   # added by stage 3
data/
  images                (n_images, S, S, 3) uint8
  generated_images      (n_generated, S, S, 3) uint8   # feedback datasets only
  image_index           (rows,) int64
  generated_image_index (rows,) int64                  # -1 for base rows
  context_tokens        (total_tokens, context_dim)    # added by stage 3
  context_offsets       (rows + 1,) int64              # added by stage 3
meta/
  split_id, is_feedback, metadata_index, sample_index, feedback_index   # per row
  caption, feedback, tuple_id                                          # per row, utf-8
```

`datasets.clevr.dataset.ClevrContextDataset` reads this layout; `datasets.build_clevr_dataset`
is the config-facing factory that mixes multiple stores with sampling ratios.

## Other files here

- `dataset.py` — dataset classes, `DistributedWeightedSampler`, `context_collate`.
- `utils.py` — chain caption rendering, transforms, zarr writers.
