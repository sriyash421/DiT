"""Evaluation metric: Gemini (via OpenRouter) edit-distance from a generated image to the ground truth."""
from verifiers.open_router import OpenRouterVerifier

EVAL_MODEL = "google/gemini-3.1-flash-lite"


def make_scorer(workers=16, max_tokens=16, timeout=120):
    return OpenRouterVerifier(model=EVAL_MODEL, temperature=0.0, max_tokens=max_tokens, workers=workers, timeout=timeout)


def score(scorer, captions, gt_images, pred_images):
    """Score each (gt image, caption, predicted image) triple; returns a distance (0-9) or None per row."""
    results = scorer.score_distance(captions, gt_images, pred_images)
    return [result.score if result.ok else None for result in results]
