"""CompBenchEval: the original T2I-CompBench faithfulness metric as an image scorer, plus a VLM-free
verifier (CompBenchFeedbackVerifier) built on top of it.

Unlike the VLM feedback verifiers in this package (which talk to a chat endpoint and return a
free-text critique / caption-mismatch *distance*, lower = better), this reproduces the per-category
metrics defined by the T2I-CompBench(++) paper and returns a faithfulness *score* in [0, 1] where
**higher = better**. It picks the best image out of N candidates (argmax over score), scores an image
for eval, and derives feedback locally — the lowest-P("yes") noun phrase is the worst-matched element,
so it becomes the fix command — with no hosted VLM.

Metric routing (paper's evaluation protocol):
  - color / shape / texture  -> Disentangled BLIP-VQA  (product of P("yes") over the prompt's
                                noun phrases, BLIP VQA "rank" inference over {yes, no})
  - non_spatial              -> CLIPScore               (image-text cosine similarity)
  - spatial / 3d_spatial /   -> UniDet detector metric  (object detection + geometric / count rule)
    numeracy
  - complex                  -> 3-in-1                   (mean of the applicable sub-scores)

UniDet needs Detectron2 + the UniDet weights, which are awkward to install. When a UniDet scorer is
not wired in (use_unidet=False or import fails), spatial/3d_spatial/numeracy fall back to BLIP-VQA on
the whole prompt and a warning is logged once. BLIP-VQA and CLIPScore are implemented natively with
HuggingFace transformers (the same BLIP capfilt-large / CLIP ViT-B-32 weights the official repo uses),
so attribute binding + non-spatial + complex are faithful out of the box.

Heavy deps (torch, transformers, optionally spacy) are imported lazily so importing this module is cheap.
"""
import re
import warnings

from verifiers.base import FeedbackVerifier, VerificationResult

# Category -> metric family. Mirrors T2I-CompBench++ examples/dataset/<category>_{train,val}.txt.
BLIP_CATEGORIES = ("color", "shape", "texture")
CLIP_CATEGORIES = ("non_spatial",)
UNIDET_CATEGORIES = ("spatial", "3d_spatial", "numeracy")
COMPLEX_CATEGORIES = ("complex",)
ALL_CATEGORIES = BLIP_CATEGORIES + CLIP_CATEGORIES + UNIDET_CATEGORIES + COMPLEX_CATEGORIES

DEFAULT_BLIP_MODEL = "Salesforce/blip-vqa-capfilt-large"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def _normalize_category(category):
    if not category:
        return "complex"
    c = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "colour": "color",
        "non_spatial_relationship": "non_spatial",
        "nonspatial": "non_spatial",
        "spatial_relationship": "spatial",
        "3d_spatial_relationship": "3d_spatial",
        "counting": "numeracy",
        "numeric": "numeracy",
        "complex_composition": "complex",
    }
    return aliases.get(c, c)


def split_noun_phrases(prompt, nlp=None):
    """Disentangle a prompt into the noun phrases BLIP-VQA is questioned on.

    With spaCy: the sentence's noun chunks. Without it (dependency-light default): split on the
    conjunctions/prepositions CompBench attribute prompts use ("a green bench and a blue bowl" ->
    ["a green bench", "a blue bowl"]). Always returns at least the whole prompt.
    """
    prompt = str(prompt).strip().rstrip(".")
    if not prompt:
        return [prompt]
    if nlp is not None:
        chunks = [chunk.text.strip() for chunk in nlp(prompt).noun_chunks]
        chunks = [c for c in chunks if c]
        if chunks:
            return chunks
    parts = re.split(r"\s+(?:and|with|next to|on the (?:left|right|top|bottom)|near|beside)\s+", prompt)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [prompt]


class CompBenchEval:
    """Scores (prompt, image, category) triples with the T2I-CompBench metric. Higher = more faithful."""

    def __init__(
        self,
        device=None,
        blip_model=DEFAULT_BLIP_MODEL,
        clip_model=DEFAULT_CLIP_MODEL,
        use_spacy=True,
        use_unidet=False,
        unidet_scorer=None,
        dtype="float16",
    ):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.float16)
        if self.device == "cpu":
            self.dtype = torch.float32
        self.blip_model_id = blip_model
        self.clip_model_id = clip_model
        self.use_spacy = bool(use_spacy)
        self.use_unidet = bool(use_unidet)
        self._unidet = unidet_scorer  # object with .score(prompt, image, category)->float, or None
        # Lazy handles.
        self._blip = None
        self._blip_proc = None
        self._clip = None
        self._clip_proc = None
        self._nlp = None
        self._warned_unidet = False

    # ------------------------------------------------------------------ loaders
    def _ensure_blip(self):
        if self._blip is not None:
            return
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self._blip_proc = BlipProcessor.from_pretrained(self.blip_model_id)
        self._blip = (
            BlipForQuestionAnswering.from_pretrained(self.blip_model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    def _ensure_clip(self):
        if self._clip is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._clip_proc = CLIPProcessor.from_pretrained(self.clip_model_id)
        self._clip = (
            CLIPModel.from_pretrained(self.clip_model_id, torch_dtype=self.dtype).to(self.device).eval()
        )

    def _ensure_nlp(self):
        if not self.use_spacy or self._nlp is not None:
            return
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.use_spacy = False
            self._nlp = None

    # ------------------------------------------------------------------ metrics
    @property
    def torch_no_grad(self):
        return self.torch.no_grad

    def _blip_yes_prob(self, image, question):
        """BLIP-VQA 'rank' inference over {yes, no}: P(yes) = softmax of the two answers'
        sequence log-likelihoods. Matches the official disentangled BLIP-VQA scoring."""
        self._ensure_blip()
        torch = self.torch
        logps = []
        for answer in ("yes", "no"):
            inputs = self._blip_proc(image, question, return_tensors="pt").to(self.device)
            labels = self._blip_proc.tokenizer(answer, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                out = self._blip(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs["pixel_values"].to(self.dtype),
                    labels=labels,
                )
            # loss is mean NLL over answer tokens; total log-prob = -loss * num_tokens.
            n_tokens = int((labels != self._blip_proc.tokenizer.pad_token_id).sum().item()) or labels.shape[1]
            logps.append(-float(out.loss.item()) * n_tokens)
        yes, no = logps
        m = max(yes, no)
        import math

        return math.exp(yes - m) / (math.exp(yes - m) + math.exp(no - m))

    def _blip_phrase_probs(self, prompt, image):
        """P('yes') that each of the prompt's noun phrases is present, one BLIP-VQA pass per phrase."""
        self._ensure_nlp()
        probs = []
        for phrase in split_noun_phrases(prompt, self._nlp):
            question = phrase if phrase.endswith("?") else f"{phrase}?"
            probs.append((phrase, self._blip_yes_prob(image, question)))
        return probs

    def _blip_vqa_disentangled(self, prompt, image):
        score = 1.0
        for _, prob in self._blip_phrase_probs(prompt, image):
            score *= prob
        return float(score)

    def _clip_score(self, prompt, image):
        self._ensure_clip()
        torch = self.torch
        inputs = self._clip_proc(text=[prompt], images=[image], return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.no_grad():
            img_emb = self._clip.get_image_features(pixel_values=inputs["pixel_values"])
            txt_emb = self._clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cos = float((img_emb * txt_emb).sum(dim=-1).item())
        return max(0.0, cos)  # monotonic for argmax; clamped like CompBench's max(cos, 0)

    def _unidet_score(self, prompt, image, category):
        if self._unidet is not None:
            return float(self._unidet.score(prompt, image, category))
        if not self._warned_unidet:
            warnings.warn(
                f"CompBenchEval: UniDet scorer unavailable; category '{category}' falls back to "
                "whole-prompt BLIP-VQA. Install Detectron2 + UniDet and pass unidet_scorer for the "
                "paper-exact spatial/numeracy metric.",
                stacklevel=2,
            )
            self._warned_unidet = True
        return self._blip_yes_prob(image, prompt if prompt.endswith("?") else f"{prompt}?")

    # ------------------------------------------------------------------ public API
    def score_one(self, prompt, image, category):
        """Faithfulness score in [0, 1] for a single (prompt, image) under its category's metric."""
        category = _normalize_category(category)
        image = image.convert("RGB") if hasattr(image, "convert") else image
        if category in BLIP_CATEGORIES:
            return self._blip_vqa_disentangled(prompt, image)
        if category in CLIP_CATEGORIES:
            return self._clip_score(prompt, image)
        if category in UNIDET_CATEGORIES:
            return self._unidet_score(prompt, image, category)
        # complex -> 3-in-1: mean of the sub-metrics we can compute.
        subs = [self._blip_vqa_disentangled(prompt, image), self._clip_score(prompt, image)]
        if self.use_unidet or self._unidet is not None:
            subs.append(self._unidet_score(prompt, image, "spatial"))
        return float(sum(subs) / len(subs))

    def score(self, prompts, images, categories=None):
        """Vectorized over a list of (prompt, image[, category]) -> list[float] faithfulness scores."""
        if categories is None:
            categories = ["complex"] * len(prompts)
        assert len(prompts) == len(images) == len(categories)
        return [self.score_one(p, im, c) for p, im, c in zip(prompts, images, categories)]

    def select_best(self, prompt, images, category):
        """Score all candidate images for one prompt; return (best_index, best_score, all_scores)."""
        scores = [self.score_one(prompt, im, category) for im in images]
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        return best_index, scores[best_index], scores

    def feedback(self, prompt, image, exclude=(), threshold=0.5):
        """Feedback from BLIP-VQA: the lowest-P('yes') noun phrase not already addressed becomes a
        fix command. Returns 'no update' when the worst remaining phrase already clears `threshold`."""
        image = image.convert("RGB") if hasattr(image, "convert") else image
        exclude = {str(item).strip().lower() for item in (exclude or ())}
        ranked = sorted(self._blip_phrase_probs(prompt, image), key=lambda pair: pair[1])
        for phrase, prob in ranked:
            command = f"add {phrase.strip()}"
            if command.lower() in exclude:
                continue
            return command if prob < threshold else "no update"
        return "no update"


class CompBenchFeedbackVerifier(FeedbackVerifier):
    """VLM-free CompBench verifier: BLIP-VQA gives the lowest-prob phrase as feedback, and the
    faithfulness score (higher = better) as the reward. Drop-in for the hosted-VLM verifier in the
    CompBench feedback and eval paths — no chat endpoint, only the local BLIP/CLIP metric models.
    """

    workers = 1

    def __init__(self, category="complex", **eval_kwargs):
        self.category = category
        self.eval = CompBenchEval(**eval_kwargs)

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        return VerificationResult(
            ok=True, feedback=self.eval.feedback(caption, attempt_image, exclude=feedback_history)
        )

    def _distance_row(self, caption, gt_image, attempt_image):
        return VerificationResult(
            ok=True, score=float(self.eval.score_one(caption, attempt_image, self.category))
        )
