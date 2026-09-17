"""Feedback chains drawn with the project's own trace renderer.

Reuses algorithms/eval.py::render_adaptive_trace -- the strip graphic already used for
eval/adaptive_traces -- rather than inventing a second layout: attempt --critique arrow-->
attempt --> ... with the ground truth at the far right. Borders are coloured green/exact,
red/not via the exact_flags argument added for this.

Examples are picked one per outcome category, and the counts over all 200 held-out chains are
printed so it is clear whether the shown rows are typical.
"""
import argparse, json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/mmfs1/gscratch/socialrl/sriyash/DiT/.claude/worktrees/g6-curriculum")
from algorithms.eval import render_adaptive_trace  # noqa: E402


def categorise(tr, steps=None):
    """Categorise on the attempts that will actually be DISPLAYED. Categorising on the full
    8-attempt chain while drawing only the first 4 produced rows whose label contradicted the
    images (a chain marked "repaired" showing a final red border)."""
    ex = [a["exact"] for a in tr["attempts"][:steps]]
    if not any(ex):
        return "never"
    if ex[0] and ex[-1]:
        return "correct_held"
    if any(ex) and not ex[-1]:
        return "lost"
    if not ex[0] and len(ex) > 1 and ex[1]:
        return "repaired_at_2"
    return "repaired_late"


TITLES = {
    "repaired_at_2": "repaired by the first critique",
    "repaired_late": "repaired later in the chain",
    "correct_held":  "correct immediately, then held",
    "lost":          "was correct, then broken again",
    "never":         "never repaired",
}
ORDER = ["repaired_at_2", "repaired_late", "correct_held", "lost", "never"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4, help="attempts to draw per chain")
    ap.add_argument("--tile", type=int, default=150)
    ap.add_argument("--categories", default=None,
                    help="comma-separated subset of the outcome categories to draw, in order; "
                         "default draws every category that has an example")
    ap.add_argument("--title", default=None, help="banner drawn above the rows")
    a = ap.parse_args()

    traces = json.load(open(a.traces))
    buckets = {}
    for tr in traces:
        buckets.setdefault(categorise(tr, a.steps), []).append(tr)
    total = len(traces)
    print(f"{total} chains:")
    for k in ORDER:
        n = len(buckets.get(k, []))
        print(f"  {k:<15s} {n:>4d}  ({n/total:5.1%})  {TITLES[k]}")

    wanted = [c.strip() for c in a.categories.split(",")] if a.categories else ORDER
    unknown = [c for c in wanted if c not in TITLES]
    assert not unknown, f"unknown categories {unknown}; pick from {ORDER}"

    rows, labels = [], []
    for cat in wanted:
        pool = buckets.get(cat)
        if not pool:
            continue
        tr = sorted(pool, key=lambda t: len(t["caption"]))[0]
        att = tr["attempts"][:a.steps]
        # render_adaptive_trace expects each entry to carry the feedback that PRODUCED it, so the
        # critique computed from attempt k conditions attempt k+1.
        entries = [{"image": Image.open(att[0]["path"]).convert("RGB"), "feedback_used": ""}]
        for k in range(1, len(att)):
            entries.append({"image": Image.open(att[k]["path"]).convert("RGB"),
                            "feedback_used": att[k - 1]["feedback"] or "no update"})
        strip = render_adaptive_trace(
            entries, Image.open(tr["gt_path"]).convert("RGB"), tr["caption"],
            tile=a.tile, exact_flags=[x["exact"] for x in att])
        rows.append(strip)
        labels.append(f"{TITLES[cat]}   ({len(pool)}/{total} chains)")

    font = ImageFont.load_default()
    band = 22
    head = 24 if a.title else 0
    width = max(r.width for r in rows)
    height = sum(r.height + band for r in rows) + 10 * (len(rows) - 1) + head
    out = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    y = 0
    if a.title:
        draw.rectangle((0, 0, width, head - 1), fill=(225, 228, 235))
        draw.text((10, 7), a.title, fill=(20, 20, 20), font=font)
        y = head
    for strip, label in zip(rows, labels):
        draw.rectangle((0, y, width, y + band - 1), fill=(242, 242, 245))
        draw.text((10, y + 6), label, fill=(40, 40, 40), font=font)
        out.paste(strip, (0, y + band))
        y += band + strip.height + 10
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print("saved ->", a.out, out.size)


if __name__ == "__main__":
    main()
