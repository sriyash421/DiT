"""Evaluation metric: Gemini (via OpenRouter) counts how many caption attributes a generated image misses."""
from verifiers.open_router import OpenRouterVerifier

EVAL_MODEL = "google/gemini-3.1-flash-lite"


def make_scorer(workers=16, max_tokens=1024, timeout=120, backend="vlm", **compbench_kwargs):
    """Build the eval scorer (higher = better). backend='vlm' -> the structured CLEVR check (-distance);
    backend='compbench' -> the CompBenchEval faithfulness score."""
    if backend == "compbench":
        from verifiers.compbench import CompBenchFeedbackVerifier
        return CompBenchFeedbackVerifier(**compbench_kwargs)
    # max_tokens is generous: the distance prompt returns a bulleted list of mismatches, not a single int.
    return OpenRouterVerifier(model=EVAL_MODEL, temperature=0.0, max_tokens=max_tokens, workers=workers, timeout=timeout)


def scorer_from_eval_cfg(eval_cfg, **kwargs):
    """Pick the eval scorer from a trainer.eval config node (all scorers are higher-is-better):
    CompBench runs (use_compbench_scorer) get the CompBench faithfulness distance on the color
    category; everything else gets the CLEVR structured VLM distance."""
    use_compbench = bool(eval_cfg.get("use_compbench_scorer", False)) if eval_cfg is not None else False
    if use_compbench:
        return make_scorer(backend="compbench", category="color", **kwargs)
    return make_scorer(**kwargs)


def score(scorer, captions, gt_images, pred_images):
    """Score each (caption, predicted image) pair; returns a reward (higher = better) or None per row.

    gt_images is accepted for signature compatibility but unused — scoring is caption-grounded.
    """
    results = scorer.score_distance(captions, gt_images, pred_images)
    return [result.score if result.ok else None for result in results]
