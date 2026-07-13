"""Evaluation metric: Gemini (via OpenRouter) counts how many caption attributes a generated image misses."""
from verifiers.open_router import OpenRouterVerifier

EVAL_MODEL = "google/gemini-3.1-flash-lite"


def make_scorer(workers=16, max_tokens=256, timeout=120):
    # max_tokens is generous: the distance prompt returns a bulleted list of mismatches, not a single int.
    return OpenRouterVerifier(model=EVAL_MODEL, temperature=0.0, max_tokens=max_tokens, workers=workers, timeout=timeout)


def score(scorer, captions, gt_images, pred_images):
    """Score each (caption, predicted image) pair; returns a caption-mismatch count (>=0) or None per row.

    gt_images is accepted for signature compatibility but unused — scoring is caption-grounded.
    """
    results = scorer.score_distance(captions, gt_images, pred_images)
    return [result.score if result.ok else None for result in results]
