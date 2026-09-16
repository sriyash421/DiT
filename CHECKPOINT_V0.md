# CHECKPOINT v0 — reproducing the `clevr_g6` on-policy feedback report

Tag **`v0`** marks the code that produces every number in the report as of 2026-09-16:
the data, the three models, the evaluation and the figures. Large artifacts are **not** stored in
git; this file records where they live and how to rebuild them.

**Report:** https://wandb.ai/sriyash-uw-team/clevr_g6/reports/clevr_g6-—-data-scale,-memorisation,-and-on-policy-feedback--VmlldzoxNzk0MTIyMw==
**wandb:** entity `sriyash-uw-team`, project `clevr_g6`, figures logged to run `m2hasqdr`.

---

## 1. What v0 is, and one honesty note about its provenance

v0 branches from `0a92651`, which is the **earliest commit containing the control run's exact
config, the g6 dataset configs and the cell-aware verifier**. Everything after it on
`worktree-g6-anchor` (anchored repair, arms A/B, the DDP fix) is deliberately excluded, as are the
curriculum and capacity-ablation *runs*.

The honest caveat: the control run launched on 2026-09-15 00:29 from the **main checkout with
uncommitted changes**, before `0a92651` existed. `0a92651` is the commit into which those changes
were first recorded. It also carries curriculum machinery (position weights, `chains_per_prompt`,
LoRA dropout) that the control did not use — all of it opt-in and disabled by
`configs/train_on_policy_g6_tiny.yaml`, so the control's loss path is the pre-curriculum one.
Between `0a92651` and the anchor work only `datasets/rollouts.py` changed; the configs and the
verifier are byte-identical.

So: v0 reproduces the control, but it is a *reconstruction* of the control's code state, not a
commit that was checked out when the control ran.

## 2. Environment

| | |
|---|---|
| training / evaluation | `/mmfs1/gscratch/socialrl/sriyash/DiT/.venv-omni/bin/python` |
| plotting | `/mmfs1/gscratch/socialrl/sriyash/DiT/.venv-unidet/bin/python` — the only env with `seaborn`, which the house plot style needs |
| `wandb_workspaces` | not in any venv; vendored at `.../jobs/<id>/tmp/pylibs`, installed with `pip install --target`. Report builds need it on `PYTHONPATH` |
| env vars | `HF_HOME=/gscratch/scrubbed/sriyash/hf`, `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false` |

**Do not modify `.venv-omni`.** Compute nodes have no outbound network, so `pip install` fails
there; vendor into a `--target` directory and set `PYTHONPATH`.

## 3. Data

CLEVR renders come from a separate generator, not this repo:

| | path | size |
|---|---|---|
| render spec | `/mmfs1/gscratch/socialrl/sriyash/clevr_generation/image_generation/data/` (`combos_g6.json`, `cells_g6.json`, `base_scene.blend`) | — |
| renderer | `.../image_generation/render_images.py`, `collect_scenes.py` (Blender) | — |
| raw renders | `/gscratch/socialrl/sriyash/clevr_g6/images` (10,000 PNG), `.../scenes` (10,200 JSON) | — |

**Dataset:** 10,000 renders, 256×256, orthographic. 2–6 objects placed in **6 fixed grid cells**;
7 colours × 3 shapes. The caption names the objects **and** their cells, in ascending cell order,
so one caption essentially determines one image:

```
objects: purple cube, brown sphere, blue cube, purple sphere. cells: 0, 2, 3, 5.
```

Build the zarrs (`repro/data/`), in this order:

```bash
python repro/data/build_g6_zarr.py        # -> clevr_g6 (8,000 train / 2,000 test) + clevr_g6_small (1,000)
python repro/data/build_g6_tiny_zarr.py   # -> clevr_g6_tiny (100 train / same 2,000 test)
```

| zarr | path | size | splits |
|---|---|---|---|
| full | `/gscratch/socialrl/sriyash/clevr_g6/data.zarr` | 1.3 G | train 8,000 / test 2,000 |
| small | `/gscratch/socialrl/sriyash/clevr_g6_small/data.zarr` | 376 M | train 1,000 |
| **tiny** | `/gscratch/socialrl/sriyash/clevr_g6_tiny/data.zarr` | 264 M | **train 100** / test 2,000 |

Splits are **nested and leakage-free**: the 100 ⊂ the 1,000 ⊂ the 8,000, and the 2,000 test rows
are identical across all three. `build_g6_tiny_zarr.py` asserts train/test caption disjointness at
build time. **Every reported number uses the tiny zarr.**

## 4. Verifier

VLM-free. Grounding DINO for boxes + a CLIP shape probe + HSV colour, each box assigned to its
nearest of the 6 cells (30 px tolerance).

```bash
python repro/data/retrain_shape_probe_g6.py   # -> /gscratch/scrubbed/sriyash/models/clevr_shape_probe_g6.pt (523 KB)
python repro/data/validate_cellaware.py       # sanity: cell assignment on real renders
```

The shipped probe misreads cylinders as cubes at this object size (shape 0.873 vs 1.000 on held-out
g6 crops), which is why it is retrained. Config: `configs/verifier/clevr_detector_g6.yaml` —
`cell_centres` are measured from ground truth (std 0.00 px, 72 px apart), `cell_tol: 30.0`,
`max_edits: null` (emit every error in one critique).

**Scoring.** Matching is cell-keyed. A matched object earns `0.5·(shape right) + 0.5·(colour
right)`, halved if blurry; `score = credit / (objects asked for + surplus objects)`. `exact` is
`score == 1.0`, equivalently the critique being `"no update"` — verified equivalent in code and at
1.0000 agreement over 22,400 images.

**Ceilings** (the verifier's own reading of *real* renders — a perfect generator cannot beat these):

```bash
python repro/eval/ceiling_g6.py                                    # held-out 200
python repro/eval/ceiling_g6.py --split train --count 100 --out .../ceiling_g6_train.json
```

| | exact | score | presence | shape | colour | precision |
|---|---|---|---|---|---|---|
| held-out 200 | **0.975** | 0.997 | 1.000 | 1.000 | 0.994 | 1.000 |
| train 100 | **0.960** | 0.995 | 1.000 | 1.000 | 0.991 | 1.000 |

Colour is the only component the detector ever misreads; that alone explains the exact-scene gap.

## 5. Models

All three are OmniGen-v1 (3.76B), 256×256, batch 32, lr 5e-5. **Checkpoints are not in git.**

| name in the report | run dir | steps used | how trained | size |
|---|---|---|---|---|
| **undertrained base** | `/gscratch/socialrl/sriyash/OmniGen-clevr-g6-tiny2k/clevr_g6_tiny2k` | **250** | full FT (`lora_finetune: false`), 100 images | 170 G |
| **offline model** | `/gscratch/socialrl/sriyash/OmniGen-clevr-g6-tiny/clevr_g6_tiny` | 2500 / 5000 / 7500 / 10000 / 15000 | full FT, same 100 images, `lr_warmup_steps: 1000` | 128 G |
| **in-context model** (the control arm) | `/gscratch/scrubbed/sriyash/omni-clevr-g6-on-policy/on_policy_g6_tiny_d250` | **5000** | LoRA r128 on the frozen base @250 | 40 G |

The two full-FT runs differ only in `lr_warmup_steps` (1000 vs 100) and `max_train_steps`
(20000 vs 2000); immaterial at ≥2500 steps but worth stating.

**Train the in-context model** (2 GPUs; `global_batch_size` is divided by world size, so the
effective batch is 32 at any rank count):

```bash
torchrun --nnodes=1 --nproc_per_node=2 train.py --config-name train_on_policy_g6_tiny
# resume after preemption:
#   ckpt=<run>/resume.bkp trainer.resume=<run>/resume.bkp
```

Method: the model generates, the verifier critiques **in language**, the model regenerates
conditioned on `[caption, attempt₁, critique₁, …]`, and the records are trained with ordinary flow
matching toward ground truth. **The verifier's score never enters the loss** — it only writes the
context. One `x_T` per chain is held fixed across its attempts, so the only thing changing between
attempts is the critique. Chains run their full length even after success, so the model sees
"already correct, no update" transitions. Config: chain length 4, 5,000 steps, LoRA r128 on
`qkv_proj, o_proj, gate_up_proj, down_proj`.

## 6. Evaluation protocol

**One protocol for every reported number:** 200 held-out captions (`prompt_seed 0`), chains of 8,
`history_window 3`, same verifier throughout. Train-split numbers use all 100 train captions.

```bash
REPO_DIR=<repo> RUN_DIR=<run> STEP=<step> TAG=<tag> OUTNAME=<name>.json \
  sbatch repro/jobs/job_seqiid_generic.sh      # held-out 200
  sbatch repro/jobs/job_seqiid_train.sh        # train 100
```

Both wrap `repro/eval/seq_vs_iid_passk.py`, which runs two regimes on the same captions:

- **seq** — the feedback chain: generate → critique → regenerate, `x_T` held fixed, context capped
  to the last 3 attempts.
- **iid** — `k` fresh samples with the verifier selecting the best (best-of-k). Reported with the
  unbiased estimator `1 − C(n−c,k)/C(n,k)`.

Both spend ≈k generations and ≈k verifier calls, so **the budgets are matched**.

Useful flags: `--save_traces N --trace_dir DIR` (chain images + critiques, for the trace figures),
`--constant_feedback "no update"` and `--shuffle_feedback` (placebo controls), `--exclude_n`
(draw prompts disjoint from the reported set; **never used for any reported number**).

Outputs land in `/gscratch/socialrl/sriyash/clevr_g6_bon/`:
`seqiid/` (held-out), `seqiid_train/` (train), `traces/`, `figs/`, `ceiling_g6*.json`.

## 7. Figures and report

```bash
sbatch repro/jobs/job_rebuild_figs.sh "<report url>"   # rebuild every figure, then republish
```

Runs plotting under `.venv-unidet` and the report build under `.venv-omni`. Passing the report URL
updates that report **in place**; omitting it creates a new one. A figure whose eval has not landed
leaves a visible "Figure pending" placeholder rather than vanishing.

| figure | script |
|---|---|
| fig6 pass@k, fig7 perf@k, fig13/14 train | `repro/figures/plot_two_tier.py` |
| fig10 checkpoint ladder | `repro/figures/fig_ckpt_ladder.py` |
| fig11 offline trajectory | `repro/figures/fig_baseline_curve.py` |
| fig15 training variants | `repro/figures/fig_methods.py` |
| fig8/fig16 chain traces | `repro/figures/fig_traces.py` |
| fig9 equal-budget bars | `repro/figures/fig_three_way.py` |

Report text lives in `repro/report/report_sections_v6.py`; `build_report_v6.py` assembles it.

## 8. Numbers a correct rerun must reproduce

Held-out pass@k (exact scenes), 200 captions, ± standard error ≈ 0.035:

| arm | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| undertrained base @250 · best-of-k | 0.054 | 0.100 | 0.176 | 0.285 |
| offline @2500 · best-of-k (**best of five**) | 0.224 | 0.287 | 0.343 | **0.390** |
| in-context @5000 · **feedback chain** | 0.350 | 0.490 | 0.540 | **0.560** |
| in-context @5000 · best-of-k | 0.338 | 0.432 | 0.511 | 0.575 |

Offline ladder at k=8: 0.390 / 0.385 / 0.380 / 0.375 / 0.375 for 2.5k–15k — **flat**, all paired
|t| < 2, while train pass@8 stays 0.95. Train captions, in-context chain: pass@8 **0.910** against
a 0.960 ceiling.

Chain outcomes over 200 held-out chains, attempt 1 vs attempt 4: 65 held / **22 repaired** /
**5 broken** / 108 wrong at both ends (91 never exact, 17 briefly exact then lost).

Reproducibility observed: three independent evals of the same checkpoint gave chain pass@8 =
0.560 / 0.560 / 0.565, so run-to-run noise (~0.005) is far below the standard error.

## 9. Known issues at v0

Found by adversarial audit; none changes a reported number, all are worth fixing before building on
this.

1. **"More training" is ruled out; "training of this shape" is not.** The offline ladder is the
   control for extra optimisation and it settles the generic version of the objection: six times
   the gradient steps on the identical 100 images (2.5k -> 15k) never approaches the in-context
   model (best-of-k at k=1: 0.240 vs 0.338) and is flat to declining at k=8.

   What remains unseparated is narrower: the in-context model is **LoRA r128 on a frozen base**
   while the baselines are **full fine-tunes**, and it is trained on contexts that contain previous
   images. Either could regularise or help at 100 images independently of what the critiques say.
   The control that isolates this is base@250 + 5,000 LoRA steps on the same 100 images with the
   same chain structure but no critique content -- not a generic "train the baseline longer" run,
   which has already been done.
2. **The chain does not beat sampling from the same model at k=8** (0.560 vs 0.575, paired
   t = −0.46). It wins at k=2 (+0.058, t = +2.12). The defensible claim is that the chain is worth
   one to two extra samples at small budgets.
3. **The model trains on critiques written by the scorer that then grades it**, including its 30 px
   cell tolerance and colour thresholds. The offline baseline has no such exposure. Confirming the
   gap survives an independent scorer needs a second verifier.
4. **The best offline checkpoint is chosen on the same 200 captions that are reported.** This
   favours the baseline, so it is conservative — but say so.
5. `algorithms/eval.py::adaptive_eval` (the in-training diagnostic, not the reported eval) drew
   fresh noise per chain step at v0, and its eval set is 32 prompts with a different seed each
   time. Fixed after v0; the reported pass@k was always noise-bound.
6. `history_instruction` renders an **empty** critique as "no changes needed, the image already
   matches the prompt" — so a detector failure would silently become an affirmative claim that a
   wrong image is correct. Never fired (0 of 1,600 critiques), but it is a trapdoor.
7. Figure error bars are unpaired per-arm standard errors, while the deltas quoted in the text are
   paired. Paired SEs are tighter.

## 10. Deliberately not in v0

The curriculum arm, the capacity ablation (r32 attention-only), and anchored single-step repair
(arms A and B) — all on `worktree-g6-anchor` and later branches. The curriculum *code* is present
at this commit but disabled by the control's config.
