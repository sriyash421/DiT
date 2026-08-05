"""Evaluation metric: a VLM (via OpenRouter) counts how many caption attributes a generated image misses."""
from verifiers.open_router import OpenRouterVerifier

EVAL_MODEL = "z-ai/glm-4.6v"


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
    # Open-vocab Grounding DINO numeracy scorer (score = dense). Used for the on-policy TTS curve and
    # eval/distance_* metrics so they measure numeracy directly instead of the CLEVR VLM distance.
    # VLM-free CLEVR (color+shape) scorer: GDino detect + learned CLIP-probe shape + HSV color.
    if eval_cfg is not None and bool(eval_cfg.get("use_clevr_detector_scorer", False)):
        from verifiers.detector_clevr import ClevrDetectorVerifier, DEFAULT_PROBE
        return ClevrDetectorVerifier(
            device="cuda",
            probe_path=eval_cfg.get("detector_probe_path", DEFAULT_PROBE),
            use_probe=bool(eval_cfg.get("detector_use_probe", True)),
            use_owl=bool(eval_cfg.get("detector_use_owl", False)),  # scipy-free by default (.venv-omni has no scipy)
        )
    if eval_cfg is not None and bool(eval_cfg.get("use_gdino_scorer", False)):
        from verifiers.gdino_feedback import GDinoNumeracyFeedbackVerifier
        return GDinoNumeracyFeedbackVerifier(device="cuda")
    use_compbench = bool(eval_cfg.get("use_compbench_scorer", False)) if eval_cfg is not None else False
    if use_compbench:
        category = eval_cfg.get("category", "color") if eval_cfg is not None else "color"
        return make_scorer(backend="compbench", category=category, **kwargs)
    return make_scorer(**kwargs)


def score(scorer, captions, gt_images, pred_images):
    """Score each (caption, predicted image) pair; returns a reward (higher = better) or None per row.

    gt_images is accepted for signature compatibility but unused — scoring is caption-grounded.
    """
    results = scorer.score_distance(captions, gt_images, pred_images)
    return [result.score if result.ok else None for result in results]
