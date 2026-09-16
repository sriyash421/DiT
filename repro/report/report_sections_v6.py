"""Sections 4-10. The plots carry the content; text is lists and tables only.

Naming, for readers outside the project:
  undertrained base model = the 100-image model at step 250 (where feedback training started)
  offline model           = the same 100 images trained normally, no feedback, many checkpoints
  in-context model        = the undertrained base + 5,000 steps of feedback training
  anchored repair model   = single-step repair; the draft is pinned to the frozen base
  feedback chain          = generate, critique, regenerate, k times (noise held fixed)
  independent samples     = k fresh samples, no feedback (best-of-k)

Sections whose runs are still going carry a PENDING marker in place of numbers, never a guess.
"""

PENDING = "> _Numbers pending — run still going. Titles and axes are final._"

STRIP_NOTE = ("The lower strip maxes over attempts **independently per metric**, so no single "
              "image achieves all of those values at once; the main panel demands exactly that "
              "from one image.")


def sections(wr, fig):
    return [
        # ------------------------------------------------------------------ section 4
        wr.H1("4 · How the verifier scores an image"),
        wr.MarkdownBlock(
            "No VLM. A detector enumerates the scene — Grounding DINO for boxes, a CLIP probe for "
            "shape, HSV for colour — and each box is assigned to its nearest of the 6 grid cells "
            "(tolerance 30 px). The caption names the objects **and their cells**, so scoring is a "
            "code-level diff, not a judgement.\n\n"
            "**Matching is cell-keyed.** The object the caption puts in cell 3 is compared against "
            "whatever was detected *in cell 3*:\n\n"
            "| what is in that cell | outcome |\n|---|---|\n"
            "| right colour **and** shape | full credit |\n"
            "| right colour, wrong shape | half credit |\n"
            "| right shape, wrong colour | half credit |\n"
            "| wrong on both, or nothing | **missing** — and the object found there becomes surplus |\n\n"
            "So an object in the wrong cell is punished twice: missing where it should be, surplus "
            "where it is."),
        wr.MarkdownBlock(
            "**The score.** Each matched object earns `0.5 × (shape right) + 0.5 × (colour right)`, "
            "halved again if blurry. Then:\n\n"
            "```\n"
            "score = total credit / (objects asked for + surplus objects)\n"
            "```\n\n"
            "Surplus objects grow the **denominator**, so they cost you even when everything asked "
            "for is present — 3 of 3 right plus one spurious object scores 3/4, not 1.0.\n\n"
            "The other five metrics are the parts of that number, all over the objects asked for:\n\n"
            "| metric | definition |\n|---|---|\n"
            "| **presence** | fraction matched at all (recall) |\n"
            "| **shape** | fraction with the right shape |\n"
            "| **colour** | fraction with the right colour |\n"
            "| **quality** | fraction of matched objects that are crisp, not blurry |\n"
            "| **precision** | matched / (matched + surplus) |\n"
            "| **exact** | **score is exactly 1.0** |"),
        wr.MarkdownBlock(
            "**Why exact is so much lower than the components.** `exact` is a conjunction over the "
            "whole scene, the others are partial credit. Measured over the 1,600 graded images in "
            "the in-context model's chains:\n\n"
            "| | value |\n|---|---|\n"
            "| mean score, all images | 0.771 |\n"
            "| mean score of **exact** images | 1.000 |\n"
            "| mean score of **non-exact** images | **0.587** |\n"
            "| `exact` ⟺ `score == 1.0` | agreement **1.000** |\n\n"
            "Scenes that fail are still mostly right. With per-object accuracy ~0.77 and 2–6 "
            "objects per scene, the chance every object is right runs from 0.59 down to 0.21 — and "
            "the observed exact rate, 0.445, sits inside that range.\n\n"
            "**Worked example.** Caption asks for 4 objects; the model gets 3 perfect and one with "
            "the right shape but the wrong colour:\n\n"
            "| | |\n|---|---|\n"
            "| credit | 3×1.0 + 1×0.5 = 3.5 |\n"
            "| score | 3.5 / 4 = **0.875** |\n"
            "| presence | 1.000 · shape 1.000 · **colour 0.750** · precision 1.000 |\n"
            "| exact | **no** |\n\n"
            "> One wrong colour out of four moves every component barely at all, and moves `exact` "
            "from 1 to 0. That is the whole gap between the two rows of every figure below."),

        # ------------------------------------------------------------------ section 5
        wr.H1("5 · Training a model to use feedback"),
        wr.MarkdownBlock(
            "The verifier critiques each attempt **in language**. The next attempt is conditioned "
            "on `[caption, attempt₁, critique₁, …]`.\n\n"
            "| | |\n|---|---|\n"
            "| start | **undertrained base** — 100 images, 250 steps |\n"
            "| data | the **same 100 images**, no new supervision |\n"
            "| trainable | fresh LoRA r128 on a frozen base |\n"
            "| chain | 4 attempts, run in full even after success |\n"
            "| noise | one x_T per chain, fixed across its attempts |\n"
            "| total | 5,000 steps |\n\n"
            "- The verifier's **score never enters the loss** — it only writes the language context.\n"
            "- The objective is ordinary flow matching toward the ground-truth image.\n"
            "- Fixed noise makes each step an **edit**: only the critique changes between attempts."),
        fig("fig5", "Ground truth · the model's attempt · the verifier's critique."),

        # ------------------------------------------------------------------ section 6
        wr.H1("6 · Held-out — what does feedback buy?"),
        wr.MarkdownBlock(
            "One budget of *k* generations, spent two ways:\n\n"
            "- **feedback chain** — generate → critique → regenerate, *k* times\n"
            "- **independent samples** — *k* fresh samples, the verifier picking the best (best-of-k)\n\n""Both spend k generations and about k verifier calls, so the budgets are matched.\n\n"
            "200 held-out captions. The baseline is the **offline model at its best checkpoint**."),
        fig("fig6", "pass@k. Error bars are the standard error across captions; the dotted line in "
                    "each panel is that metric's ceiling on real renders. " + STRIP_NOTE),
        fig("fig7", "performance@k — attempt k alone. " + STRIP_NOTE),
        wr.MarkdownBlock(
            "> Only the in-context model's chain **improves** with attempts. Models never trained "
            "for feedback are flat or decaying given the same critiques."),

        wr.H2("At equal budget"),
        fig("fig9", "Three strategies, same budget. Left: verifier score of the image you keep · "
                    "Right: how often it is exactly right."),
        wr.MarkdownBlock(
            "All deltas below are **exact-scene pass@k**, paired per caption, against the offline "
            "model's best checkpoint.\n\n"
            "| k | 1 | 2 | 4 | 8 |\n|---|---|---|---|---|\n"
            "| in-context **chain** − offline | +0.11 | **+0.20** | +0.18 | +0.17 |\n"
            "| in-context **iid** − offline | +0.11 | +0.15 | +0.17 | **+0.19** |\n"
            "| chain − its **own** iid | +0.01 | +0.06 | +0.03 | **−0.02** |\n\n"
            "- A converged model handed the identical critiques improves **not at all** — its chain "
            "is flat at 0.23.\n"
            "- The feedback-trained model beats the offline model at every budget in **both** "
            "regimes (paired t = 4.2 and 5.6 at k=8).\n"
            "- **The chain itself is worth roughly one to two extra samples at k ≤ 4, and nothing "
            "at k = 8** (chain − own iid at k=8: −0.015, paired t = −0.46). Most of the margin over "
            "the offline model comes from the in-context model being a better one-shot generator "
            "(iid@1 0.338 vs 0.224), not from the critiques at eval time."),

        wr.H2("Example chains"),
        fig("fig8", "HELD-OUT captions. Green = exact, red = not. Each arrow carries the critique "
                    "computed from the image on its left."),
        fig("fig16", "The same graphic on TRAINING captions — targets the model saw during "
                     "training. Two rows: a chain the first critique repairs, and one it never "
                     "repairs."),
        wr.MarkdownBlock(
            "> On training captions the model is already right at attempt 1 four times in five, so "
            "most chains have nothing to repair. The rows worth seeing are the two shown."),

        # ------------------------------------------------------------------ section 7
        wr.H1("7 · Is the baseline just under-trained?"),
        wr.MarkdownBlock(
            "The offline model is evaluated along its **whole run** — 2.5k to 15k steps on the same "
            "100 images — so the comparison is against its best form, not a convenient checkpoint."),
        fig("fig10", "Every offline checkpoint's best-of-k, light to dark with training step, "
                     "against the in-context feedback chain in purple."),
        fig("fig11", "The offline trajectory at k=8, with the in-context chain as a reference band."),
        wr.MarkdownBlock(
            "| offline steps | 2,500 | 5,000 | 7,500 | 10,000 | 15,000 |\n"
            "|---|---|---|---|---|---|\n"
            "| held-out pass@8 | **0.390** | 0.385 | 0.380 | 0.375 | 0.375 |\n"
            "| train pass@8 | 0.960 | 0.950 | 0.950 | 0.950 | 0.950 |\n\n"
            "> **Six times the training buys nothing on held-out.** Every paired difference across "
            "the ladder has |t| < 2, so the curve is flat rather than declining — while train "
            "performance sits pinned at 0.95. The baseline in every other figure is the best of "
            "these five, chosen on the same captions we report, which favours the baseline."),

        # ------------------------------------------------------------------ section 8
        wr.H1("8 · The same picture on training captions"),
        wr.MarkdownBlock(
            "All 100 captions the models were trained on, same protocol. This separates **what was "
            "learned** from **what transfers**."),
        fig("fig13", "pass@k on training captions."),
        fig("fig14", "performance@k on training captions."),
        wr.MarkdownBlock(
            "| in-context model, training captions | exact |\n|---|---|\n"
            "| attempt 1 | 0.810 |\n| chain of 8 | 0.910 |\n\n"
            "> The draft became **strong on its own training captions** during the run — which is "
            "why the supply of errors to learn repair from dried up. That motivates section 10."),

        # ------------------------------------------------------------------ section 9
        wr.H1("9 · What the chains actually do"),
        wr.MarkdownBlock(
            "Per-caption outcomes over the 200 held-out chains, comparing attempt 1 with attempt 4 "
            "(one definition throughout, so the rows partition the 200):\n\n"
            "| outcome | captions |\n|---|---|\n"
            "| right at attempt 1, still right at 4 | 65 |\n"
            "| wrong at 1, **repaired** by 4 | **22** |\n"
            "| right at attempt 1, then **broken** | **5** |\n"
            "| wrong at both ends | 108 |\n"
            "| &nbsp;&nbsp;— of which never exact at any attempt | 91 |\n"
            "| &nbsp;&nbsp;— of which briefly exact, then lost | 17 |\n\n"
            "Over the full 8 attempts: **31 fixed, 5 broken** — net **+26**.\n\n"
            "> Repair clearly outweighs damage — 22 fixed against 5 broken. But the dominant "
            "outcome is **no repair at all**: 91 of 200 captions are never exact at any attempt.\n\n"
            "> The 17 chains that were briefly right and then lost it are the interesting cell — "
            "the model found the correct image and talked itself out of it."),

        wr.H2("Training variants"),
        wr.MarkdownBlock(
            "Solid = feedback chain, dashed = independent samples, for each configuration.\n\n"
            "| variant | chain | LoRA | steps | score@1 | score@8 | exact@8 |\n"
            "|---|---|---|---|---|---|---|\n"
            "| **in-context** | 4 | r128 | 5,000 | 0.713 | 0.845 | **0.560** |\n"
            "| **curriculum** | 2→3→4 | r32 attn-only | 1,400 | 0.475 | 0.611 | 0.175 |\n"
            "| **capacity ablation** | 4 | r32 attn-only | 1,000 | _pending_ | _pending_ | _pending_ |"),
        fig("fig15", "On-policy configurations, same eval protocol. NOT matched on training steps "
                     "or LoRA capacity — different configs stopped at different points, so this is "
                     "a comparison of what was run, not a controlled ablation."),
        wr.MarkdownBlock(
            "> The curriculum arm is far behind, but it ran **1,400 steps at r32** against the "
            "in-context model's **5,000 at r128** — most of that gap is budget, not curriculum.\n\n"
            "> The one thing the gap does not explain: for the curriculum model, **independent "
            "sampling beats its own feedback chain at every k** (0.773 vs 0.611 at k=8). The "
            "in-context model is the opposite way round at low k. Whatever the curriculum taught, "
            "it was not better use of a critique.\n\n"
            "> The capacity-ablation arm was never evaluated — its run died on a disk-quota error "
            "at step 1,000, not a method failure. Its eval is running now."),

        # ------------------------------------------------------------------ section 10
        wr.H1("10 · Anchored single-step repair"),
        wr.MarkdownBlock(
            "If the draft keeps improving, the model runs out of its own mistakes to practise on. "
            "This run **stops training the draft**.\n\n"
            "| | |\n|---|---|\n"
            "| chain | 2 — draft, then one repair |\n"
            "| draft step | **no ground truth**; held softly at the frozen base |\n"
            "| repair step | supervised to ground truth, 25% caption dropout |\n"
            "| effect | error supply stays stationary; all capacity goes into the edit |\n\n"
            "Caption dropout matters because the caption alone determines the target — without it "
            "the model memorises caption→image at the repair step and ignores the critique."),
        fig("fig12", "Anchored repair against the same baselines."),
        wr.MarkdownBlock(PENDING),

        # ------------------------------------------------------------------ limits
        wr.H1("Limits"),
        wr.MarkdownBlock(
            "- The in-context model had **more total training** than any single offline checkpoint; "
            "section 7 is what addresses this.\n"
            "- The offline model's best checkpoint is picked on the **same 200 captions** the "
            "figures report — that favours the baseline, so it does not inflate the result.\n"
            "- The context holds the previous image **and** the critique; nothing here separates "
            "them, so we cannot say the model is reading the words.\n"
            "- Chains were trained at length 4, evaluated to 8.\n"
            "- The chain reuses one noise draw by design, denying it the diversity independent "
            "sampling has.\n"
            "- The scorer is a detector stack, not ground truth. Its own ceiling on real renders is "
            "in Setup, and colour is the only component it ever misreads."),
    ]
