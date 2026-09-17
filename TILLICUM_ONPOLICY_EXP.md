# Rejection-sampled on-policy feedback training — run instructions for tillicum

You are running **one experiment** on 8× H200: on-policy feedback training with a cap on how much
of the rollout buffer is allowed to be degenerate. Everything needed is on this branch
(`onpolicy_exp`). The artifacts are **not** in git; §2 says how to fetch them.

Read §1 before changing anything — the experiment is a one-variable change against an existing
control, and the value is entirely in keeping it that way.

---

## 1. What this tests, and why

**The finding that motivates it.** A previous run ("the control": 100 CLEVR images, LoRA r128 on a
frozen base, chain length 4, 5,000 steps) trained the model to generate an image, receive a
code-written critique, and regenerate. It worked — held-out exact-scene pass@8 went 0.390 → 0.560
against the best no-feedback baseline — but it plateaued.

The cause was measured directly from that run's 100 saved rollout buffers. A chain is **degenerate**
when its *first* attempt is already exact: every later row in it is then conditioned on the critique
`"no update"`, so it teaches *copy the image you were shown*, not *repair it*. The share of repair
rows in that state over training:

| outer loop | 1 | 11 | 21 | 41 | 71 | 100 |
|---|---|---|---|---|---|---|
| % conditioned on `"no update"` | 9.7 | 37 | 54 | 80 | 86 | **87.7** |

By outer 40 the buffer is ~80% copy-lesson and stays there. Roughly the **last 3,000 of 5,000
gradient steps** trained on a buffer where about one row in eight carried a real edit instruction.
That also explains a failure seen at eval: of 200 held-out chains, 22 took an image that was already
correct and broke it.

**The intervention.** Keep sampling chains until the buffer is full of `samples` chains of which at
most `max_no_update_frac` are degenerate. The buffer stays **full** — rejection costs generation,
not training signal.

**Why not just filter them out?** Filtering shrinks the buffer, which silently changes epochs per
rollout at fixed `updates_per_rollout`. Rejection keeps buffer size, batch size and epoch count
identical to the control, so the only thing that differs is composition.

**Do not set the cap to 0.** The model still has to learn to hold an image that is already correct;
that is exactly what the 22-broken-chains failure punishes. 0.25 keeps the lesson present at a
quarter of the buffer while leaving three-quarters informative.

### The comparison that matters

| | control (already run) | this arm |
|---|---|---|
| base | full-FT 100-image model @ step 250 | **same** |
| data | the same 100 images | **same** |
| adapter | LoRA r128 on qkv/o/gate_up/down | **same** |
| chain | length 4, fixed x_T per chain | **same** |
| optimiser | batch 32, lr 5e-5 fixed, wd 0, 5,000 steps | **same** |
| buffer composition | uncontrolled → 88% degenerate | **capped at 25%** |

Held-out numbers the control produced, which this arm is measured against (200 captions,
`prompt_seed 0`, window 3, chains of 8, exact-scene pass@k):

| arm | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| offline full FT, best of 5 checkpoints | 0.224 | 0.287 | 0.343 | 0.390 |
| no-feedback LoRA (same setup, no critiques) | 0.304 | 0.380 | 0.450 | 0.500 |
| **control, feedback chain** | 0.350 | 0.490 | 0.540 | **0.560** |
| control, best-of-k | 0.338 | 0.432 | 0.511 | 0.575 |

**Success looks like** beating 0.560 at k≥2, and ideally raising the chain above the control's own
best-of-k (0.575), which is the bar the control failed to clear.

---

## 2. Fetch the artifacts (they are not in git)

Both are private HuggingFace repos under `sriyash421`. You need a token with read access.

```bash
export HF_TOKEN=<token>          # or: hf auth login
export HF_HOME=<somewhere with ~40GB free>

# 1) the dataset: 100 train + 200 held-out images, and their captions
hf download sriyash421/clevr-g6-tiny-v0 --repo-type dataset --local-dir ./artifacts/images

# 2) the base checkpoint the LoRA trains on top of  (22.8 GB)
hf download sriyash421/onpolicy-distill-v0 \
    base_undertrained_step0000250.pt --local-dir ./artifacts/checkpoints
```

Training reads a **zarr**, not the loose PNGs. It is published in the same dataset repo:

```bash
# 3) the zarr the trainer actually reads (100 train / 2,000 held-out)
hf download sriyash421/clevr-g6-tiny-v0 --repo-type dataset \
    --include 'data.zarr/*' --local-dir ./artifacts
# -> ./artifacts/data.zarr

# 4) the verifier's shape probe (523 KB) -- see §3 for why the stock one will not do
hf download sriyash421/clevr-g6-tiny-v0 --repo-type dataset \
    verifier/clevr_shape_probe_g6.pt --local-dir ./artifacts
# -> ./artifacts/verifier/clevr_shape_probe_g6.pt
```

Everything the run needs is in those two repos; nothing has to be copied off hyak.

The loose PNGs under `images/` are for inspection and for pinning the eval set; they are **not**
what training or evaluation reads. If you ever need to rebuild the zarr from scratch,
`repro/data/build_g6_tiny_zarr.py` does it, but it needs the 10,000 raw CLEVR renders, which are
not published — download the zarr instead.

Note the zarr must be written by **zarr 2.x**. An environment with zarr 3.x can read it but cannot
write the v2 schema, which is why the builder and the trainer historically needed different envs.

Verify the eval set is the right one before trusting any number:

```
sha256 of the ordered captions, first 16 hex
  train  fa0845d22203cf2b
  val    432cefd7055dc827
```

### Point the configs at what you fetched — with env vars, not edits

Every cluster-specific path is `${oc.env:VAR,<hyak default>}`, so **do not edit the config files**;
export these instead. Editing them creates a diff that makes this arm hard to compare against the
control, which is the whole point of the run.

**The filesystems differ between clusters.** hyak's scratch is `/gscratch/scrubbed/sriyash`;
tillicum's is **`/gpfs/scrubbed/sriyash`**. Every default below is a hyak path and will not exist
here, so all five must be set.

```bash
export G6_DATA=/gpfs/scrubbed/sriyash/artifacts/data.zarr
export G6_BASE_CKPT=/gpfs/scrubbed/sriyash/artifacts/checkpoints/base_undertrained_step0000250.pt
export G6_PROBE=/gpfs/scrubbed/sriyash/artifacts/verifier/clevr_shape_probe_g6.pt
export G6_RESULTS=/gpfs/scrubbed/sriyash/runs/onpolicy        # ~50 GB
export ROLLOUT_DIR=/gpfs/scrubbed/sriyash/rollouts/g6_reject  # fast scratch, ~5 GB
```

| var | what it points at | if it is wrong |
|---|---|---|
| `G6_DATA` | the zarr the trainer reads | trains on the wrong images, silently |
| `G6_BASE_CKPT` | the frozen base the LoRA sits on | **silently trains on raw OmniGen-v1** — the single most dangerous mistake here |
| `G6_PROBE` | the retrained CLEVR shape probe | every score is wrong; the shipped probe reads cylinders as cubes |
| `G6_RESULTS` | checkpoints and logs | a full filesystem kills the run mid-eval |
| `ROLLOUT_DIR` | rollout buffers | slow scratch makes generation the bottleneck |

Confirm they resolved before launching — this prints the paths hydra will actually use:

```bash
python -c "
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir='$PWD/configs', version_base=None):
    c = compose(config_name='train_on_policy_g6_reject')
for k, v in (('zarr', c.dataset.datasets[0].path), ('base_ckpt', c.model.base_ckpt),
             ('results', c.results_dir), ('probe', c.verifier.probe_path),
             ('rollouts', c.trainer.rollout.storage_dir)):
    print(f'{k:<10}', v)
"
```

If any line still says `/gscratch`, the env var did not take and you are about to run against paths
that do not exist on this cluster.

## 3. The verifier

VLM-free: Grounding DINO for boxes + a CLIP shape probe + HSV colour, each box snapped to its
nearest of 6 fixed grid cells (30 px tolerance). Config: `configs/verifier/clevr_detector_g6.yaml`.

It needs `clevr_shape_probe_g6.pt` (523 KB), fetched in §2 from the dataset repo under
`verifier/`. The stock CLEVR probe reads cylinders as cubes at this object size (shape 0.873 vs
1.000 on held-out crops), so the retrained one is **required** — with the stock probe every critique
and every score is wrong, and wrong in a way that still looks plausible. `G6_PROBE` must point at
the fetched file. (`repro/data/retrain_shape_probe_g6.py` can rebuild it, but there is no reason
to.)

Sanity check before a long run: the verifier should call ~97.5% of *real* renders exact. If it is
far off, the probe or the cell centres are wrong.

## 4. Run it

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --config-name train_on_policy_g6_reject
# resume after preemption or a time limit:
#   ckpt=<results_dir>/on_policy_g6_reject/resume.bkp \
#   trainer.resume=<results_dir>/on_policy_g6_reject/resume.bkp
```

**Always give the job `--requeue` and a generous `--time`.** On hyak this arm's sibling died at
exactly 08:00:05 with no error at all — a wall-clock kill, no requeue, run over. Non-preemptable
does not mean untimed.

### Watch these, in order of importance

| signal | healthy | wrong |
|---|---|---|
| `rollout/degenerate_frac` | **≤ 0.25**, the cap | drifting above ⇒ oversample budget exhausted |
| `rollout/oversample` | ~1× early, rising to ~4–6× | pinned at 8× ⇒ raise `max_oversample` or relax the cap |
| `rollout/draft_exact` | rises (a good draft is wanted here) | — |
| `rollout/repair_exact` | rises **faster** than draft | flat ⇒ the repair signal is still not landing |
| `success_rate` | 1.000 | < 1 ⇒ verifier failures, check the probe |

`draft_exact` / `repair_exact` are the verifier's **exact-scene rate on the 100 TRAIN captions**,
logged every rollout. They are not a loss and not the partial score, and they overstate held-out
ability badly — on the control, 0.83 on train sat against 0.35 held-out. Use them for **trend**, and
judge the arm only by the held-out eval in §6.

---

## 5. Scaling to 8× H200

The defaults in the config are tuned for **4× L40S (48 GB)**. H200s are 141 GB and you have twice as
many, so several things can move — but **two must not**, or the comparison against the control is
lost.

### Do not change these

| key | value | why |
|---|---|---|
| `trainer.global_batch_size` | **32** | It is divided by world size, so the effective batch is 32 at any rank count. Raising it changes the optimisation and the arm stops being comparable to the control. |
| `trainer.lr`, `weight_decay`, `max_train_steps`, `updates_per_rollout` | as-is | Same reason. The control is 5,000 steps at lr 5e-5. |

### Safe to change

**Turn gradient checkpointing OFF.** `model.gradient_checkpointing: false`. It was only ever enabled
because `rollout_loss` OOM'd on a 48 GB L40S — a record carries up to 3 prior attempt images (768
image tokens) plus repeated captions. At 141 GB it is unnecessary, and disabling it removes a
recompute of every forward. Numerically identical. This is the single best speedup available.

**Raise `trainer.eval.gen_batch`.** Currently 8 on the anchor configs, and the control leaves it at
the default 32. `adaptive_eval` generates its whole chunk in one batch; at 141 GB you can use 32 or
more. On hyak this exact setting at 4 ranks caused a CUDA OOM 44 minutes into a run, because three
peer ranks each park ~2.8 GB of NCCL buffers on GPU 0 — at 8 ranks that overhead is ~7× one rank's
worth, so keep an eye on GPU 0 specifically.

**`trainer.rollout.batch_size`** must be divisible by world size. At 8 ranks, 32 → 4/rank, which is
small. 128 (16/rank) or 256 (32/rank) both fit easily in 141 GB.

> Temper your expectations here. On hyak, doubling this from 16→32 per rank produced **+1.1%** —
> 380 s → 376 s per collect — because generation was already at 77–97% GPU utilisation and is
> **compute-bound, not memory-bound**. More GPUs help nearly linearly; a bigger batch per GPU barely
> does. Do not spend a restart on it expecting more.

### The one thing 8 ranks genuinely changes

At `global_batch_size: 32` and 8 ranks the **local** update batch is 4 rows. With chain length 4 the
buffer is 25% drafts / 75% repairs, so a local batch of 4 is single-class reasonably often. That is
survivable — the loss path evaluates both terms unconditionally and zero-scales whichever is empty —
but a rank whose batch is all-one-class contributes nothing to the other term that step.

An earlier version of this code *skipped* the empty term instead, which made ranks issue different
numbers of DDP collectives and killed two runs with a 10-minute NCCL watchdog timeout and no Python
traceback:

```
Rank 0] WorkNCCL(SeqNum=2605, OpType=ALLREDUCE, ...) timed out
Rank 1] WorkNCCL(SeqNum=2605, OpType=BROADCAST, ...) timed out
```

That is fixed on this branch. **If you ever add a branch in `update()` that depends on batch
contents, every rank must still run the same number of forwards.** `repro/tests/test_rank_uniform.py`
guards this statically — run it after touching the loss.

### Suggested 8× H200 starting point

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --config-name train_on_policy_g6_reject \
    model.gradient_checkpointing=false \
    trainer.rollout.batch_size=128 \
    trainer.eval.gen_batch=32 \
    trainer.dataloader.num_workers=4
```

Rough cost. On 4× L40S a collect of 400 generations took ~184 s. Eight H200s should be
~3–4× faster per generation-batch, and rejection multiplies generation by ~1× early rising to ~6×
late, averaging ~3–4×. Those roughly cancel: expect **wall clock in the same ballpark as the control
run**, i.e. order a day for 5,000 steps, not a week. Measure the first few loops and extrapolate
rather than trusting this estimate.

---

## 6. Evaluate

**Use the same protocol as every published number, or the result is not comparable:** 200 held-out
captions, `prompt_seed 0`, chains of 8, `history_window 3`, the same verifier.

```bash
python repro/eval/seq_vs_iid_passk.py \
  --run_dir <results_dir>/on_policy_g6_reject --step 5000 \
  --split test --num_prompts 200 --n 8 --batch_size 16 \
  --prompt_seed 0 --history_window 3 --regimes seq,iid \
  --tag reject_5000 --out bd_reject_5000.json
```

Then compare against the control with `repro/analysis/three_claims.py`, or plot with
`repro/figures/plot_two_tier.py` (it takes `--nofeedback` and `--placebo` too).

Two eval notes that have caused mistakes here:

- **pass@k vs perf@k.** `pass@k` is the union over the first k attempts and is monotone; `perf@k` is
  attempt k *alone* and can fall. Quoting one against the other inflates a result — the control's
  0.560 (pass@8) and 0.475 (perf@8) are not interchangeable.
- **Window.** Report window 3 to match every existing arm. Window 1 is arguably more principled for
  short chains, but switching it moves arms in opposite directions, so pick on principle and report
  both if it matters.

## 7. What is on this branch

```
algorithms/on_policy.py        rejection sampling in RolloutCollector.collect()
configs/train_on_policy_g6_reject.yaml   this arm  (control + the cap)
configs/train_on_policy_g6_tiny.yaml     the control, for diffing
repro/eval/                    the pass@k driver used for every published number
repro/figures/                 every figure in the report
repro/analysis/                the scripts behind the claims
repro/data/                    zarr builders, shape-probe retraining, verifier validation
repro/tests/test_rank_uniform.py         the DDP guard described in §5
```

## 8. Known traps, all of which have already bitten this project

1. **`--requeue` on every job**, whatever the partition. A time-limit kill without it ends the run.
2. **Disk.** A run died with `OSError: Disk quota exceeded` while writing an eval grid. Rejection
   deletes rejected chains' files, so peak usage is about one buffer — but check free space first.
3. **Don't skip a loss term per-rank.** §5.
4. **The probe.** Wrong probe ⇒ silently wrong critiques and scores, no error.
5. **Checkpoint selection.** Report a checkpoint fixed in advance (the final one) or apply the same
   best-of-N selection to every arm. Picking the best checkpoint for the new arm only would be
   selection bias in its favour.
