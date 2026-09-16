"""Section-4 figure: what the on-policy trainer actually sees.

Each row is one training prompt from the SAME 100 images the base model was trained on:
  ground truth | the D@250 model's attempt | the caption and the verifier's critique.

The point of the figure is that the critique is specific and actionable -- it names the object, the
cell it is in, and the cell it belongs in -- so "condition the next attempt on this text" is a
well-posed learning problem rather than a vague reward.
"""
import json, sys, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
from plot_style import apply_house_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

SRC = "/gscratch/socialrl/sriyash/clevr_g6_bon/feedback/d250_train_feedback.json"
OUT = "/gscratch/socialrl/sriyash/clevr_g6_bon/figs/fig5_feedback_examples.png"
# one example per critique type, so the figure shows the full command vocabulary
WANT = ["missing", "move", "extra-replace", "shape", "extra-remove"]


def pretty_caption(cap):
    """Wrap the object list so it stays inside its column instead of running off the figure."""
    objs = cap.split("objects:", 1)[1].split(". cells:")[0].strip()
    cells = cap.split("cells:", 1)[1].rstrip(". ").strip()
    lines = textwrap.wrap(f"objects: {objs}", 56, subsequent_indent="         ")
    return "\n".join(lines + [f"cells:   {cells}"])


def main():
    rows = json.load(open(SRC))
    picked, seen = [], set()
    for rule in WANT:
        for r in rows:
            if r["rule"] == rule and r["pred"] not in seen:
                picked.append(r); seen.add(r["pred"]); break
    apply_house_style()
    n = len(picked)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 2.05 * n),
                             gridspec_kw={"width_ratios": [1, 1, 2.45]})
    for i, r in enumerate(picked):
        for j, (path, title) in enumerate(((r["gt"], "ground truth"),
                                           (r["pred"], "model attempt"))):
            ax = axes[i, j]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.75"); sp.set_linewidth(0.8)
            if i == 0:
                ax.set_title(title, fontsize=10, color="0.25", pad=6)
        ax = axes[i, 2]
        ax.axis("off")
        if i == 0:
            ax.set_title("prompt  →  verifier critique", fontsize=10, color="0.25", pad=6)
        cap_text = pretty_caption(r["caption"])
        ax.text(0.0, 0.94, cap_text, va="top", ha="left",
                fontsize=8.0, color="0.38", family="monospace", transform=ax.transAxes)
        # start the critique just under the caption, whatever height the caption wrapped to
        y = 0.94 - 0.135 * (cap_text.count("\n") + 1) - 0.08
        body = "\n".join(textwrap.wrap(r["feedback"], 56))
        ax.text(0.0, y, body, va="top", ha="left", fontsize=8.8, color="#1b3a5c",
                family="monospace", transform=ax.transAxes)
        ax.text(0.0, 0.03, f"score {r['score']:.2f}", va="bottom", ha="left",
                fontsize=7.6, color="0.55", transform=ax.transAxes)
    fig.suptitle("On-policy feedback data — D@250 attempts on its own 100 training images",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    print("rules shown:", [r["rule"] for r in picked])
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
