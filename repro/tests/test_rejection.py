"""Rejection sampling: does it actually cap the degenerate share, and is it off by default?

Runs the real RolloutCollector against a stub model and stub verifier -- no GPU, no OmniGen. The
stub reports a tunable fraction of first attempts as already exact, which is what makes a chain
degenerate.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "/gscratch/scrubbed/sriyash/onpolicy-exp")
from algorithms.on_policy import RolloutCollector  # noqa: E402


class R:
    def __init__(self, fb):
        self.feedback, self.ok, self.token_count = fb, True, 0


class StubVerifier:
    """`exact_rate` of first attempts come back "no update" -- i.e. degenerate chains."""

    def __init__(self, exact_rate):
        self.exact_rate = exact_rate
        self.calls = 0

    def verify(self, captions, gt_images, attempts, histories):
        out = []
        for i in range(len(captions)):
            first = not histories[i]
            # Deterministic so the test is reproducible: the first `exact_rate` share of each batch
            # is "already correct".
            exact = first and (i < int(round(self.exact_rate * len(captions))))
            out.append(R("no update" if (exact or not first) else "move the red cube to cell 2"))
        self.calls += 1
        return out


class StubModel:
    def generate(self, batch, **kw):
        n = len(batch["caption"])
        from PIL import Image
        imgs = [Image.new("RGB", (8, 8)) for _ in range(n)]
        lat = [torch.zeros(2, 2) for _ in range(n)]
        return imgs, lat


class Cfg:
    num_sampling_steps, cfg_scale, ddim_eta = 1, 1.0, 0.0


def make_loader(n, batch):
    items = [{"image": torch.zeros(batch, 3, 8, 8), "caption": [f"c{i}" for i in range(batch)]}
             for i in range(n)]
    return items


def run(max_frac, exact_rate, samples=20, batch=4, length=2):
    tmp = Path(tempfile.mkdtemp())
    try:
        col = RolloutCollector(StubModel(), Cfg(), StubVerifier(exact_rate),
                               rollout_length=length, verify_last=False,
                               max_no_update_frac=max_frac, max_oversample=8.0)
        stats, _ = col.collect(make_loader(400, batch), tmp, samples, device="cpu", seed=0), None
        comp = col.last_rollout_composition
        return comp, tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("stub draft is exact 75% of the time -- the late-training regime the control hit\n")
fail = []

comp, _ = run(max_frac=None, exact_rate=0.75)
print(f"rejection OFF   : accepted {comp['accepted_chains']}, generated {comp['generated_chains']}, "
      f"degenerate {comp['degenerate_frac']:.0%}, oversample {comp['oversample']:.2f}x")
if comp["oversample"] > 1.01:
    fail.append("rejection disabled should not oversample at all")
if comp["accepted_chains"] != 20:
    fail.append(f"disabled: expected a full buffer of 20, got {comp['accepted_chains']}")

for cap in (0.25, 0.5):
    comp, _ = run(max_frac=cap, exact_rate=0.75)
    print(f"cap {cap:.0%}        : accepted {comp['accepted_chains']}, "
          f"generated {comp['generated_chains']}, degenerate {comp['degenerate_frac']:.0%}, "
          f"oversample {comp['oversample']:.2f}x, rejected {comp['degenerate_rejected']}")
    if comp["accepted_chains"] != 20:
        fail.append(f"cap {cap}: buffer not full ({comp['accepted_chains']}/20)")
    if comp["degenerate_frac"] > cap + 1e-6:
        fail.append(f"cap {cap}: degenerate share {comp['degenerate_frac']:.2f} exceeds the cap")
    if comp["oversample"] <= 1.0:
        fail.append(f"cap {cap}: expected extra sampling, got {comp['oversample']:.2f}x")

# A draft that is never exact produces no degenerate chains, so the cap must cost nothing.
comp, _ = run(max_frac=0.25, exact_rate=0.0)
print(f"\nno degenerates  : oversample {comp['oversample']:.2f}x (should be 1.00x -- cap never binds)")
if comp["oversample"] > 1.01:
    fail.append("cap should not oversample when no chain is degenerate")

print()
if fail:
    print("FAILED:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("buffer stays full, the cap holds, and the cost only appears when degenerates exist")
