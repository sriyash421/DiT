"""clevr_g6 report: setup, three data-scale claims, and the on-policy feedback experiment.

Sections 4-5 are new. Every number in them comes from ONE eval protocol -- 200 held-out captions,
prompt_seed 0, history window 3, chains of 8 -- so nothing is mixed across runs. Curriculum and
ablation arms are deliberately excluded (still training).

Style: minimal prose, tables and figures carry the content.
"""
import re
import sys
import wandb
sys.path.insert(0, "/gscratch/scrubbed/sriyash/.claude/jobs/9b095793/tmp")
import wandb_workspaces.reports.v2 as wr

ENTITY, PROJECT, RUN = "sriyash-uw-team", "clevr_g6", "m2hasqdr"

# Resolve each figure to its LATEST logged version. run.files() returns every version of a media
# file (media/images/report/<name>_<step>_<hash>.png); keeping the first match would silently pin a
# stale figure after a re-log -- exactly what happens when numbers are corrected.
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN}")
best = {}
for f in run.files():
    if not (f.name.endswith(".png") and "report/" in f.name):
        continue
    stem = f.name.split("report/")[1]
    key = stem.split("_")[0]
    m = re.search(r"_(\d+)_[0-9a-f]+\.png$", stem)
    step = int(m.group(1)) if m else -1
    if key not in best or step >= best[key][0]:
        best[key] = (step, f.url)
urls = {k: v[1] for k, v in best.items()}
print("resolved figures:", {k: best[k][0] for k in sorted(best)}, flush=True)

MISSING = []


def fig(k, caption):
    """Embed a figure; if it has not been logged yet, leave a visible placeholder in its slot.

    Silently dropping it would make a half-finished report look finished.
    """
    if k in urls:
        return wr.Image(url=urls[k], caption=caption)
    MISSING.append(k)
    return wr.MarkdownBlock(f"> _Figure pending — `{k}`: {caption}_")


def blocks_of(*items):
    return [b for b in items if b is not None]


from report_sections_v6 import sections as _sections
SECTIONS45 = _sections(wr, fig)

blocks = blocks_of(
    wr.H1("Setup"),
    wr.MarkdownBlock(
        "**`clevr_g6`** — 10,000 CLEVR renders, 256×256, orthographic camera. Each scene puts "
        "**2–6 objects** into **6 fixed grid cells**; 7 colours × 3 shapes. The caption names the "
        "objects *and* their cells, so one caption has essentially one correct image:\n\n"
        "> `objects: purple cube, brown sphere, blue cube, purple sphere. cells: 0, 2, 3, 5.`\n\n"
        "| | |\n|---|---|\n"
        "| splits | 8,000 train / 2,000 held-out, leakage-free |\n"
        "| model | OmniGen-v1 (3.76B), 256×256, batch 32, lr 5e-5 |\n"
        "| scoring | VLM-free: Grounding DINO + CLIP shape probe + HSV colour, each box assigned to "
        "its nearest cell |\n"
        "| **exact** | every cell holds the right object |\n"
        "| scorer accuracy | **0.997** mean partial score on the real renders of the same 200 "
        "held-out captions the figures report |\n"
        "| scorer ceiling on *exact* | **0.975** — it calls 97.5%% of those real renders exact, so "
        "that, not 1.0, is the top of the pass@k axis. Colour is the only component it ever misses "
        "(0.994); presence, shape and precision are exact on every real render |\n"
        "| sections 1–3 eval | 300 captions, cell-accuracy sweep |\n"
        "| **sections 4–5 eval** | **200 held-out captions**, `prompt_seed 0`, chains of 8, window 3 |\n\n"
        "Every section below uses one protocol throughout, on captions that are bit-identical "
        "across all arms. Train and held-out captions are disjoint by construction."),

    wr.H1("1 · 1,000 images vs 100 images — the gap is only on held-out"),
    fig("fig1", "Held-out captions. Left: ground truth · Middle: 1,000-image model · Right: 100-image model."),
    fig("fig2", "Unbiased pass@k over exact scenes. Left: train captions · Right: held-out."),
    wr.MarkdownBlock(
        "| captions scored | 1,000-image model | 100-image model |\n|---|---|---|\n"
        "| train | 0.80 | **0.89** |\n| held-out | **0.83** | 0.51 |\n\n"
        "Same two models, opposite ordering — the 100-image model has memorised its training set."),

    wr.H1("2 · The 100-image model at convergence"),
    fig("fig3", "100-image model, 2.5k steps. Top rows: training captions · Bottom rows: held-out."),
    wr.MarkdownBlock(
        "| | exact | count correct |\n|---|---|---|\n"
        "| its own 100 train captions | **0.950** | 1.000 |\n| held-out | **0.237** | 0.723 |"),

    wr.H1("3 · Under-fitting train looks like failing to generalise"),
    fig("fig4", "Top: 250-step checkpoint on captions it was trained on · Bottom: converged model on held-out."),
    wr.MarkdownBlock(
        "| | exact | cells correct | count correct |\n|---|---|---|---|\n"
        "| 250 steps, **train** captions | 0.170 | 0.544 | 0.690 |\n"
        "| converged, **held-out** | 0.237 | 0.662 | 0.723 |\n"
        "| converged, **train** | 0.950 | 0.985 | 1.000 |\n\n"
        "Rows 1 and 2 sit in the same regime for opposite reasons — one hasn't learned the mapping, "
        "the other learned only its 100 examples. That makes the 250-step checkpoint a "
        "matched-difficulty starting point with known-reachable headroom."),

    *SECTIONS45,
)

TITLE = "clevr_g6 — data scale, memorisation, and on-policy feedback"
DESC = ("Where the generalisation gap appears, and whether verifier feedback in the loop closes it.")

# An existing report URL updates that report IN PLACE, so adding a figure later does not leave a
# second, stale copy behind. Without it a fresh report is created.
existing = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") else None
if existing:
    rep = wr.Report.from_url(existing)
    rep.title, rep.description, rep.blocks = TITLE, DESC, blocks
    print("updating existing report in place:", existing, flush=True)
else:
    rep = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, description=DESC, blocks=blocks)
rep.save()
if MISSING:
    print("WARNING: figures not yet logged, omitted from the report:", MISSING, file=sys.stderr)
print("REPORT URL:", rep.url, flush=True)
