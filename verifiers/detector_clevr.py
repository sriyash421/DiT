"""VLM-free CLEVR (color+shape) verifier -- config key `use_clevr_detector_scorer`.

Pipeline (no API calls, all local):
  1. Grounding DINO detects object boxes (query the three shapes), cross-shape NMS to dedup, and the
     count is capped at the OWLv2 min-ensemble total (both detectors over-count; the min cancels the bias).
  2. Each box -> SHAPE via a learned MLP probe on frozen CLIP-ViT-B/32 image features (0.99 on held-out
     CLEVR; fixes GDino's cube/cylinder confusion), and COLOR via nearest CLEVR-palette HUE in HSV (0.99;
     shading-invariant, gray by low saturation), and a blurry flag from box Laplacian sharpness.
  3. The resulting scene enumeration feeds the shared enumeration_breakdown / enumeration_feedback ->
     score in [0,1] (higher=better) + one one-object repair command.

End-to-end on 100 val clevr_easy images: shape 0.99, color 0.99, exact-scene 0.91, correct-vs-mismatch
score gap 0.65. See results/bon_comparison/REPORT.md.

Interface matches FeedbackVerifier (verify / score_distance), so it drops into the eval + BoN paths.
Select it from a trainer.eval config node with `use_clevr_detector_scorer: true`."""
import colorsys

import numpy as np

from verifiers.base import (FeedbackVerifier, VerificationResult, enumeration_breakdown,
                            enumeration_feedback)

# Standard CLEVR diffuse colors (properties.json), in HSV for hue matching.
CLEVR_RGB = {"gray": (87, 87, 87), "red": (173, 35, 35), "blue": (42, 75, 215), "green": (29, 105, 20),
             "brown": (129, 74, 25), "purple": (129, 38, 192), "cyan": (41, 208, 208), "yellow": (255, 238, 51)}
CLEVR_HSV = {c: colorsys.rgb_to_hsv(*[v / 255 for v in rgb]) for c, rgb in CLEVR_RGB.items()}
CHROMATIC = [c for c in CLEVR_RGB if c != "gray"]
SHAPES = ["cube", "sphere", "cylinder"]

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_PROBE = "/gscratch/scrubbed/sriyash/models/clevr_shape_probe.pt"
GEOMWORD = [["a cube, a box"], ["a sphere, a ball"], ["a cylinder, a can, a tube"]]  # zero-shot fallback


def _hsv_color(img_np, hsv_np, box):
    """Nearest CLEVR color by hue over the box's central 30%; gray if low saturation."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img_np.shape[1]), min(y1, img_np.shape[0])
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return "gray"
    w, h = x1 - x0, y1 - y0
    hsv = hsv_np[y0 + h * 7 // 20:y1 - h * 7 // 20 or y1, x0 + w * 7 // 20:x1 - w * 7 // 20 or x1].reshape(-1, 3)
    keep = (hsv[:, 2] > 0.12) & (hsv[:, 2] < 0.97)
    px = hsv[keep] if keep.sum() >= 3 else hsv
    s_med, v_med = float(np.median(px[:, 1])), float(np.median(px[:, 2]))
    if s_med < 0.18:
        return "gray"
    ang = px[:, 0] * 2 * np.pi
    h_med = (np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang))) / (2 * np.pi)) % 1.0

    def dist(c):
        d = abs(h_med - CLEVR_HSV[c][0])
        d = min(d, 1 - d)
        return d + (0.15 * abs(v_med - CLEVR_HSV[c][2]) if c in ("brown", "yellow", "red") else 0.0)
    return min(CHROMATIC, key=dist)


def _sharpness(gray_np, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, gray_np.shape[1]), min(y1, gray_np.shape[0])
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return 0.0
    c = gray_np[y0:y1, x0:x1].astype(np.float64)
    lap = c[2:, 1:-1] + c[:-2, 1:-1] + c[1:-1, 2:] + c[1:-1, :-2] - 4 * c[1:-1, 1:-1]
    return float(lap.var())


def _make_probe(device):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)).to(device)


class ClevrDetectorVerifier(FeedbackVerifier):
    """VLM-free CLEVR color+shape verifier (Grounding DINO + learned CLIP-probe shape + HSV color).

    score_distance -> VerificationResult with .score (enumeration score in [0,1]) and .breakdown;
    verify -> VerificationResult with .feedback (one repair command). GPU-bound, so rows run
    sequentially (workers=1). Models load lazily on first call."""

    workers = 1

    def __init__(self, device="cuda", probe_path=DEFAULT_PROBE, box_threshold=0.25, text_threshold=0.25,
                 nms_iou=0.5, owlv2_threshold=0.15, blur_thresh=50.0, use_probe=True, use_owl=False):
        self.device = device
        self.probe_path = probe_path
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = float(nms_iou)
        self.owlv2_threshold = float(owlv2_threshold)
        self.blur_thresh = float(blur_thresh)
        self.use_probe = bool(use_probe)   # False -> zero-shot CLIP shape (no probe file needed)
        self.use_owl = bool(use_owl)       # False -> skip OWLv2 count cap (avoids the scipy dep)
        self._ready = False

    # ----------------------------------------------------------------- model loading
    def _lazy_init(self):
        if self._ready:
            return
        import torch
        from transformers import (AutoProcessor, CLIPModel, CLIPProcessor,
                                  GroundingDinoForObjectDetection, Owlv2ForObjectDetection, Owlv2Processor)
        self._torch = torch
        self.proc = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.gdino = GroundingDinoForObjectDetection.from_pretrained(GDINO_MODEL).to(self.device).eval()
        if self.use_owl:
            self.owl_proc = Owlv2Processor.from_pretrained(OWLV2_MODEL)
            self.owl = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL).to(self.device).eval()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        if self.use_probe:
            ck = torch.load(self.probe_path, map_location=self.device)
            self.probe = _make_probe(self.device)
            # probe was saved as a module wrapping the Sequential under `.net`; strip the prefix so it
            # loads into the bare Sequential returned by _make_probe.
            state = {(k[4:] if k.startswith("net.") else k): v for k, v in ck["state_dict"].items()}
            self.probe.load_state_dict(state)
            self.probe.eval()
            self.mu, self.sd = ck["mu"].to(self.device), ck["sd"].to(self.device)
        self._ready = True

    # ----------------------------------------------------------------- detection
    def _owl_total(self, img):
        from torchvision.ops import nms
        torch = self._torch
        total = 0
        for shape in SHAPES:
            inp = self.owl_proc(text=[[f"a photo of a {shape}"]], images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.owl(**inp)
            ts = torch.tensor([img.size[::-1]]).to(self.device)
            if hasattr(self.owl_proc, "post_process_grounded_object_detection"):
                res = self.owl_proc.post_process_grounded_object_detection(out, threshold=self.owlv2_threshold, target_sizes=ts)[0]
            else:  # transformers < 5
                res = self.owl_proc.post_process_object_detection(out, threshold=self.owlv2_threshold, target_sizes=ts)[0]
            b, s = res["boxes"].float().cpu(), res["scores"].float().cpu()
            total += 0 if len(b) == 0 else int(len(nms(b, s, self.nms_iou)))
        return total

    def _detect_boxes(self, img):
        """Cross-shape NMS -> one box per object, capped at the OWL total count."""
        from torchvision.ops import nms
        torch = self._torch
        inp = self.proc(images=img, text=". ".join(SHAPES) + ".", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gdino(**inp)
        try:
            res = self.proc.post_process_grounded_object_detection(
                out, inp["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[img.size[::-1]])[0]
        except TypeError:  # transformers < 5 uses box_threshold=
            res = self.proc.post_process_grounded_object_detection(
                out, inp["input_ids"], box_threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[img.size[::-1]])[0]
        boxes, scores = res["boxes"].float().cpu(), res["scores"].float().cpu()
        labels = [str(l) for l in res.get("text_labels", res.get("labels"))]
        gd = [next((sh for sh in SHAPES if sh in l.lower()), "cube") for l in labels]
        if len(boxes) == 0:
            return [], []
        keep = nms(boxes, scores, self.nms_iou)
        keep = keep[torch.argsort(scores[keep], descending=True)]
        if self.use_owl:
            keep = keep[:min(len(keep), self._owl_total(img))]
        return [boxes[i].tolist() for i in keep], [gd[i] for i in keep]

    def _shapes(self, img, boxes, gd_labels):
        torch = self._torch
        if not boxes:
            return []
        crops = []
        for x0, y0, x1, y1 in boxes:
            crops.append(img.crop((max(int(x0) - 4, 0), max(int(y0) - 4, 0),
                                   min(int(x1) + 4, img.width), min(int(y1) + 4, img.height))))
        flat = [t for g in GEOMWORD for t in g]
        inp = self.clip_proc(text=flat, images=crops, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.clip(**inp)
        if self.use_probe:
            feats = out.image_embeds
            feats = feats / feats.norm(dim=-1, keepdim=True)
            idx = self.probe((feats - self.mu) / self.sd).argmax(1).cpu().numpy()
        else:
            groups = np.array([ci for ci, g in enumerate(GEOMWORD) for _ in g])
            lg = out.logits_per_image.cpu().numpy()
            idx = np.stack([lg[:, groups == c].mean(1) for c in range(3)], axis=1).argmax(1)
        return [SHAPES[i] for i in idx]

    def detect(self, image):
        """Return the enumerated scene: [{color, shape, blurry, sharpness}]."""
        self._lazy_init()
        img = image.convert("RGB")
        img_np = np.asarray(img)
        hsv_np = np.asarray(img.convert("HSV"), np.float32) / 255.0
        gray_np = np.asarray(img.convert("L"))
        boxes, gd = self._detect_boxes(img)
        shapes = self._shapes(img, boxes, gd)
        seen = []
        for box, shape in zip(boxes, shapes):
            sharp = _sharpness(gray_np, box)
            seen.append({"color": _hsv_color(img_np, hsv_np, box), "shape": shape,
                         "blurry": sharp < self.blur_thresh, "sharpness": round(sharp, 1)})
        return seen

    # ----------------------------------------------------------------- FeedbackVerifier hooks
    def _score_row(self, caption, attempt_image):
        try:
            seen = self.detect(attempt_image)
        except Exception as exc:  # a detector failure shouldn't crash the whole batch
            return VerificationResult(ok=False, error=repr(exc))
        breakdown = enumeration_breakdown(caption, seen)
        if breakdown is None:
            return VerificationResult(ok=False, error="no objects parsed from caption")
        command, _rule, _score = enumeration_feedback(caption, seen)
        return VerificationResult(ok=True, feedback=command, score=breakdown["score"], breakdown=breakdown)

    def _distance_row(self, caption, gt_image, attempt_image):
        return self._score_row(caption, attempt_image)

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        return self._score_row(caption, attempt_image)
