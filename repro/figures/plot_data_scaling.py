"""clevr_stagger5 data scaling: val score vs training-set size, against the GT ceiling.

The point of the figure is that the curve is FLAT: 100 images reach the same held-out score as
8,000, so the dataset carries no data-scaling signal. eval_loss is plotted alongside because it
does move (3.4x worse at 100 imgs) -- showing that the loss gap never reaches the images.
"""
import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/helper_scripts")
from plot_style import MARKERS, apply_house_style, legend_below, magma_colors, style_axes
import matplotlib.pyplot as plt

B = Path("/gscratch/socialrl/sriyash/clevr_stagger5_bon")
CEIL = 0.9902
RUNS = [("C (100)", 100, B / "C", 2000, 0.0718),
        ("B (1,000)", 1000, B / "B", 20000, 0.0306),
        ("A (8,000)", 8000, B / "A", 20000, 0.0211)]
SLOT = {100: 0.972, 1000: 0.975, 8000: 0.975}

def load(p):
    r = {}
    for x in csv.DictReader(open(p)):
        r[(x["metric"], int(x["step"]), x["split"], int(x["k"]))] = (float(x["mean_best"]), float(x["std_err"]))
    return r

apply_house_style()
sizes, s1, e1, s8, loss, slot = [], [], [], [], [], []
for _, n, d, step, el in RUNS:
    r = load(d / "bon_curves.csv")
    sizes.append(n)
    s1.append(r[("score", step, "test", 1)][0]); e1.append(r[("score", step, "test", 1)][1])
    s8.append(r[("score", step, "test", 8)][0])
    loss.append(el); slot.append(SLOT[n])

c = magma_colors(3)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

ax.errorbar(sizes, s1, yerr=e1, color=c[0], marker=MARKERS[0], lw=1.8, capsize=3, label="val score (k=1)")
ax.plot(sizes, s8, color=c[1], marker=MARKERS[1], lw=1.8, label="val score (best-of-8)")
ax.plot(sizes, slot, color=c[2], marker=MARKERS[2], lw=1.8, label="slot-exact accuracy")
ax.axhline(CEIL, ls="--", lw=1.2, color="0.45", label=f"ground-truth ceiling ({CEIL:.3f})")
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{n:,}" for n in sizes])
ax.set_xlabel("training images"); ax.set_ylabel("held-out score")
ax.set_ylim(0.93, 1.005)
ax.set_title("Generation quality is flat in data")
style_axes(ax)

ax2.plot(sizes, loss, color=c[0], marker=MARKERS[3], lw=1.8)
ax2.set_xscale("log"); ax2.set_xticks(sizes); ax2.set_xticklabels([f"{n:,}" for n in sizes])
ax2.set_xlabel("training images"); ax2.set_ylabel("eval loss (lower better)")
ax2.set_title("...while eval loss moves 3.4x")
style_axes(ax2)

fig.suptitle("clevr_stagger5 — 80x less data costs ~0.3 points on the held-out set", fontsize=13)
legend_below(fig, handles_from=ax, ncol=4)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
out = B / "compare" / "data_scaling.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved ->", out)
