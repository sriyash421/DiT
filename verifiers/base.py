"""Verifier base class, prompt templates, parsers, and the OpenAI-compatible chat backend."""
import base64
import concurrent.futures
import io
import json
import re
import time
from dataclasses import dataclass, field

from PIL import Image


# When False, the structured distance prompt/score AND the feedback grammar ignore object ordering
# (left-to-right / front-to-back) and consider only presence, shape, color, and extra/missing objects.
USE_ORDER = False


@dataclass
class VerificationResult:
    ok: bool
    feedback: str = ""
    score: float | None = None
    token_count: int = 0
    token_usage: dict = field(default_factory=dict)
    error: str = ""
    breakdown: dict | None = None  # per-metric decomposition of score (enumeration scorer)


def pil_to_png_base64(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_data_url(image):
    return "data:image/png;base64," + pil_to_png_base64(image)


def resize_square(image, size):
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def normalize_chat_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def clean_feedback_text(text):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    marker_match = re.search(r"(?:final|answer)\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if marker_match:
        text = marker_match.group(1).strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        text = answer_match.group(1).strip()
    text = text.replace("\r", "\n").strip()
    for line in text.splitlines():
        line = line.strip(" \t-*#`>\"'")
        if line:
            text = line
            break
    text = re.sub(r"\s+", " ", text).strip(" \"'")
    return text


def build_feedback_prompt(caption, feedback_history=(), enable_thinking=False):
    past_feedback = [str(item).strip() for item in (feedback_history or ()) if str(item).strip()]
    if past_feedback:
        history_block = "Previous feedback already given:\n" + "\n".join(
            f"- {item}" for item in past_feedback
        ) + "\n\n"
        command_line = (
            "Write one short new command to fix image 2. "
            "Do not repeat a previous command unless that exact issue is still the clearest remaining error. "
        )
    else:
        history_block = ""
        command_line = "Write one short command to fix image 2. "
    order_form = (
        " or 'swap positions of <object A> and <object B>' when two objects are in the wrong order"
        if USE_ORDER else ""
    )
    order_priority = " > position/depth" if USE_ORDER else ""
    prompt = (
        "Give feedback for a CLEVR image generator. "
        "You are shown the generated image; the caption is the only source of truth.\n"
        f"Caption: {caption}\n\n"
        f"{history_block}"
        f"{command_line}"
        "Use exactly one of these forms: "
        "'fix <object>' when an object is blurry or malformed, "
        "'replace <wrong object> with <object>' when a wrong object appears instead of an expected one, "
        "'add <object>' when an expected object is missing, "
        "'fix shape of <object> to <shape>', "
        "'change color of <object> to <color>'"
        f"{order_form}. "
        "Do not give feedback about size or material. "
        "Use this priority: missing or wrong object > shape > color > extra object"
        f"{order_priority} > blurry object. "
        "Only give 'fix <object>' for a blurry object when every expected object is already present "
        "with the correct shape and color. "
        "Mention one object and one edit only. Do not use and. Do not explain. "
        "If the generated image already follows the caption, return exactly: no update. "
        "Return only the command, under 12 words."
    )
    if enable_thinking:
        prompt += " If you reason, end with exactly: FINAL: <command under 12 words>."
    return prompt


def parse_caption_gt(caption):
    """Recover the GT from a rendered CLEVR caption: the objects in scene order, and the left-to-right
    and front-to-back orderings as 1-based scene indices. Orderings are empty for single-object scenes.
    Returns (objects, gt_left_to_right, gt_front_to_back)."""
    from datasets.clevr.utils import unique_labels

    obj_match = re.search(r"objects:\s*(.*?)\.", caption)
    objects = []
    for descr in (obj_match.group(1).split(",") if obj_match else []):
        parts = descr.split()
        if len(parts) == 4:  # full_description == "<size> <color> <material> <shape>"
            objects.append({"size": parts[0], "color": parts[1], "material": parts[2], "shape": parts[3]})
        elif len(parts) == 2:  # shape+color-only captions (e.g. clevr_easy): "<color> <shape>"
            objects.append({"color": parts[0], "shape": parts[1]})
    if not objects:
        return [], [], []

    # Optional "cells: 0, 2, 3." clause (clevr_g6): bind each expected object to a grid cell, in
    # caption order. Captions without it are unaffected -- `cell` simply stays absent and every
    # downstream check falls back to the original position-blind behaviour.
    cell_match = re.search(r"cells:\s*([0-9,\s]*)", caption)
    if cell_match:
        cells = [int(tok) for tok in cell_match.group(1).replace(".", "").split(",") if tok.strip()]
        if len(cells) == len(objects):
            for obj, cell in zip(objects, cells):
                obj["cell"] = cell

    # unique_labels() needs size/material (label_candidates indexes both); only call it when an
    # order clause is actually present to resolve, so shape+color-only captions stay parseable.
    labels = unique_labels(objects) if ("horizontal:" in caption or "depth:" in caption) else []

    def order_indices(name):
        match = re.search(rf"{name}:\s*(.*?)\.", caption)
        if not match:
            return []
        return [labels.index(lbl.strip()) + 1 for lbl in match.group(1).split(",") if lbl.strip() in labels]

    return objects, order_indices("horizontal"), order_indices("depth")


def build_distance_prompt(caption):
    """Structured BLIP-VQA-style check: list the GT objects and ask for per-object per-attribute
    presence plus the two orderings, as strict JSON. Scoring happens in code (score_structured_distance)."""
    objects, _, _ = parse_caption_gt(caption)
    listing = "\n".join(
        f"{i + 1}. " + " ".join(o[k] for k in ("size", "color", "material", "shape") if k in o)
        for i, o in enumerate(objects)
    )
    order = USE_ORDER and len(objects) > 1
    fields = (
        '{"objects": [{"present": <bool>, "shape_ok": <bool>, "color_ok": <bool>, "malformed": <bool>}, '
        '... one entry per numbered object above]'
    )
    if order:
        fields += (
            ', "left_to_right": [object numbers, left to right, present ones only], '
            '"front_to_back": [object numbers, front (closest) to back, present ones only]'
        )
    fields += ', "extra_objects": ["<color> <shape> for any object in the image NOT in the list above"]'
    fields += "}"
    return (
        "You are shown one generated CLEVR image. It should contain these objects:\n"
        f"{listing}\n\n"
        "Judge the generated image against this list. For each numbered object report whether it is "
        "present, whether its shape and color match (shape_ok/color_ok), and whether it is blurry or "
        "malformed (malformed). "
        + ("Also give the left-to-right and front-to-back order of the present objects by their numbers. "
           if order else "")
        + "In extra_objects, list any object visible in the image that is NOT one of the expected "
        "objects above (empty list if none). "
        + "Reply with ONLY this JSON, no prose or code fences:\n"
        + fields
    )


def _extract_json(text):
    text = str(text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _as_int_list(value):
    out = []
    for item in value if isinstance(value, list) else []:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            pass
    return out


def levenshtein(a, b):
    """Sequence edit distance between two lists of object indices."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Higher (less negative) is better. Content dominates order (0.7 vs 0.3); within content, attribute
# severity is shape > color (size and material are too noisy to score). See parse_caption_gt for the
# GT the VLM is judged against.
_ATTR_WEIGHTS = {"shape_ok": 0.6, "color_ok": 0.4}


def score_structured_breakdown(text, caption):
    """Decompose the VLM's structured JSON report into per-metric rewards (higher = better, each <= 0).
    Returns None if the caption or JSON can't be parsed. Keys: combined (the final distance reward) plus
    the sub-metrics presence/shape/color/lr/fb and their content/order aggregates."""
    objects, gt_lr, gt_fb = parse_caption_gt(caption)
    n = len(objects)
    if n == 0:
        return None
    data = _extract_json(text)
    if data is None:
        return None
    reports = data.get("objects")
    if not isinstance(reports, list) or len(reports) != n or not all(isinstance(r, dict) for r in reports):
        return None

    penalties = []
    present = []
    shape_bad = color_bad = absent = 0
    for i, report in enumerate(reports):
        if not report.get("present", False):
            absent += 1
            penalties.append(1.0)
            continue
        present.append(i + 1)
        shape_bad += not report.get("shape_ok", False)
        color_bad += not report.get("color_ok", False)
        penalties.append(sum(w for key, w in _ATTR_WEIGHTS.items() if not report.get(key, False)))
    lr = fb = 0.0
    if USE_ORDER and len(present) >= 2 and gt_lr and gt_fb:
        present_set = set(present)
        keep = lambda seq: [x for x in seq if x in present_set]
        m = len(present)
        lr = levenshtein(keep(gt_lr), keep(_as_int_list(data.get("left_to_right")))) / m
        fb = levenshtein(keep(gt_fb), keep(_as_int_list(data.get("front_to_back")))) / m
    order = 0.5 * lr + 0.5 * fb
    p = max(len(present), 1)

    extra_list = data.get("extra_objects")
    n_extra = len(extra_list) if isinstance(extra_list, list) else 0
    # Pair each extra object with a missing expected one: that pair is a single substitution ("replace
    # the extra with the correct object"), already counted once via the missing object's 1.0 penalty.
    # Only genuinely surplus extras are charged separately, so an extra+missing pair is not double-counted.
    leftover_extra = max(0, n_extra - absent)
    denom = n + leftover_extra
    object_distance = (sum(penalties) + leftover_extra) / denom if denom else 0.0

    if USE_ORDER:
        combined = -(0.7 * object_distance + 0.3 * order)
    else:
        combined = -object_distance

    return {
        "combined": combined,
        "content": -object_distance,       # missing + wrong-shape/color + surplus-extra, substitution-paired
        "order": -order,
        "presence": -(absent / n),         # fraction of GT objects missing (raw, before pairing)
        "shape": -(shape_bad / p),         # shape errors among present objects
        "color": -(color_bad / p),         # color errors among present objects
        "extra": -(leftover_extra / denom) if denom else 0.0,  # surplus extras after substitution pairing
        "lr": -lr,                         # normalized left-to-right order edit distance
        "fb": -fb,                         # normalized front-to-back order edit distance
    }


def score_structured_distance(text, caption):
    """The final distance reward in [-1, 0] (0 = perfect). Returns None if the caption or JSON can't be
    parsed, so the caller can mark the row failed."""
    breakdown = score_structured_breakdown(text, caption)
    return None if breakdown is None else breakdown["combined"]


def build_text_feedback(caption, vlm_json):
    """v2 feedback: turn the structured distance JSON into ONE one-object/one-edit command, in code.
    Priority: replace/add (wrong or missing object) -> shape -> color -> remove extra -> order (if
    USE_ORDER) -> blurry (LAST, only once the scene is otherwise correct). Referents come from the
    caption, so they can't drift. Returns (command, rule)."""
    from datasets.clevr.utils import unique_labels

    objects, gt_lr, gt_fb = parse_caption_gt(caption)
    reports = (vlm_json or {}).get("objects")
    if not objects or not isinstance(reports, list) or len(reports) != len(objects):
        return "no update", "match"
    labels = unique_labels(objects)

    def color_shape(i):
        return f"{objects[i]['color']} {objects[i]['shape']}"

    present = [i for i, r in enumerate(reports) if isinstance(r, dict) and r.get("present")]
    missing = [i for i, r in enumerate(reports) if isinstance(r, dict) and not r.get("present")]
    extra = [str(x).strip() for x in ((vlm_json or {}).get("extra_objects") or []) if str(x).strip()]

    # 1. wrong object: an extra appears in place of a missing expected one -> replace (one substitution)
    if extra and missing:
        return f"replace the {extra[0]} with a {color_shape(missing[0])}", "extra-replace"
    # 2. missing expected object
    if missing:
        return f"add a {color_shape(missing[0])}", "missing"
    # 3. shape
    for i in present:
        if not reports[i].get("shape_ok", True):
            return f"fix shape of the {objects[i]['color']} object to {objects[i]['shape']}", "shape"
    # 4. color
    for i in present:
        if not reports[i].get("color_ok", True):
            return f"change color of the {objects[i]['shape']} to {objects[i]['color']}", "color"
    # 5. surplus extra object (no missing left to pair with) -> remove
    if extra:
        return f"remove the {extra[0]}", "extra-remove"
    # 6. order (only when enabled)
    if USE_ORDER and len(present) >= 2 and gt_lr and gt_fb:
        ps = {i + 1 for i in present}
        keep = lambda seq: [x for x in seq if x in ps]
        for axis, gt, key in (("left-to-right", gt_lr, "left_to_right"), ("front-to-back", gt_fb, "front_to_back")):
            gt_seq = keep(gt)
            vlm_seq = keep(_as_int_list((vlm_json or {}).get(key)))
            if gt_seq != vlm_seq:
                for want, have in zip(gt_seq, vlm_seq):
                    if want != have:
                        return f"swap the {axis} positions of the {labels[have - 1]} and the {labels[want - 1]}", f"order-{axis}"
    # 7. blurry / malformed -- LAST: only once every expected object is present with correct shape+color
    for i in present:
        if reports[i].get("malformed"):
            return f"fix the {color_shape(i)}", "blurry"
    return "no update", "match"


ENUM_COLORS = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
ENUM_SHAPES = {"cube", "sphere", "cylinder"}

# Caption-free enumeration: the VLM only describes what it sees; matching, the feedback command and
# the reward are all derived in code (enumeration_feedback). Size/material are requested but unused
# downstream -- committing to them forces per-object scrutiny and measurably improves the color/shape
# accuracy that IS kept (verifier benchmark, results/verifier_bench: 78% vs 75% without them).
ENUMERATION_PROMPT = (
    "List EVERY distinct object you can see in the image, one per line, in exactly this format:\n"
    "<size> <color> <material> <shape> | <quality>\n"
    "where size is small or large; color is one of gray, red, blue, green, brown, purple, cyan, yellow; "
    "material is metal (shiny) or rubber (matte); shape is cube, sphere or cylinder -- if a shape is "
    "distorted, give the closest one; quality is 'ok' if the object is crisp and well-formed, or 'blurry' "
    "if it is deformed, smeared or half-formed. Look closely -- objects can be small or partly occluded; "
    "list each one. Do not count reflections or shadows. Do not compare with anything or judge the image; "
    "only describe it. Output only the list, nothing else."
)


def parse_enumeration(text):
    """Parse the enumeration reply into dicts (color/shape/blurry). Lines without a valid color AND
    shape are dropped, so nothing outside the CLEVR vocabulary can enter the diff."""
    seen = []
    for line in str(text).splitlines():
        words = re.findall(r"[a-z]+", line.lower())
        color = next((w for w in words if w in ENUM_COLORS), None)
        shape = next((w for w in words if w in ENUM_SHAPES), None)
        if color and shape:
            seen.append({"color": color, "shape": shape, "blurry": "blurry" in words})
    return seen


# Enumeration scoring: a matched expected object earns SHAPE_WEIGHT for the right shape and COLOR_WEIGHT
# for the right color (summing to 1.0). The scene score is the mean per-object credit over the EXPECTED
# (caption) objects, so it lies in [0, 1] -- 1.0 = every expected object present with correct shape and
# color. Missing objects contribute 0. A matched object the VLM flagged blurry/malformed keeps only
# BLUR_KEEP of its credit, so blurriness is a soft penalty that only bites once shape+color are right.
SHAPE_WEIGHT = 0.5
COLOR_WEIGHT = 0.5
BLUR_KEEP = 0.5


def match_enumeration(objects, seen):
    """Greedily match each expected object to an enumerated one: exact (color+shape) first, then
    color-only (wrong shape), then shape-only (wrong color); the rest are missing. Returns a per-expected
    list of dicts {kind, blurry, seen} (kind in exact/shape_bad/color_bad/missing) and the leftover
    (surplus) enumerated objects."""
    pool = [dict(s) for s in seen]

    # Cell-keyed matching: when the caption names cells and the detector reports them, an object is
    # only "the" expected object if it is in the RIGHT cell. A misplaced object then falls out as
    # missing-at-its-cell plus a surplus elsewhere, so placement is penalised by the existing
    # scoring with no extra metric keys.
    if objects and all("cell" in o for o in objects) and any(s.get("cell") is not None for s in seen):
        by_cell = {s["cell"]: dict(s) for s in pool if s.get("cell") is not None}
        matches = []
        for o in objects:
            s = by_cell.pop(o["cell"], None)
            if s is None:
                kind = "missing"
            elif s["color"] == o["color"] and s["shape"] == o["shape"]:
                kind = "exact"
            elif s["color"] == o["color"]:
                kind = "shape_bad"
            elif s["shape"] == o["shape"]:
                kind = "color_bad"
            else:
                kind = "missing"          # wrong colour AND wrong shape -> not this object at all
            matches.append({"kind": kind, "blurry": bool(s.get("blurry")) if s else False,
                            "seen": s if kind != "missing" else None})
            if kind == "missing" and s is not None:
                by_cell[o["cell"]] = s     # keep it as surplus
        leftovers = list(by_cell.values()) + [dict(s) for s in pool if s.get("cell") is None]
        return matches, leftovers

    def take(pred):
        for k, s in enumerate(pool):
            if pred(s):
                return pool.pop(k)
        return None

    matches = [None] * len(objects)
    for kind, pred in (
        ("exact", lambda o, s: s["color"] == o["color"] and s["shape"] == o["shape"]),
        ("shape_bad", lambda o, s: s["color"] == o["color"]),   # right color, wrong shape
        ("color_bad", lambda o, s: s["shape"] == o["shape"]),   # right shape, wrong color
    ):
        for i, o in enumerate(objects):
            if matches[i] is not None:
                continue
            m = take(lambda s, o=o: pred(o, s))
            if m is not None:
                matches[i] = {"kind": kind, "blurry": bool(m.get("blurry")), "seen": m}
    for i in range(len(objects)):
        if matches[i] is None:
            matches[i] = {"kind": "missing", "blurry": False, "seen": None}
    return matches, pool


def enumeration_breakdown(caption, seen):
    """Score the enumerated scene against the caption in code (higher = better, all in [0, 1]).
    Returns None if the caption has no objects, else a dict:
      score     -- the reward: per-object credit (0.5 shape + 0.5 color, blur-discounted) summed over
                   MATCHED objects, divided by (#expected + #surplus-extra). Mirrors the CompBench
                   overall score, where credit is normalized by what was asked for and extra detections
                   cost you -- here every surplus object grows the denominator, so 3/3 correct with one
                   spurious object scores 3/4, not 1.0.
      presence  -- fraction of expected objects that were matched at all (recall)
      shape     -- fraction of expected objects with the correct shape
      color     -- fraction of expected objects with the correct color
      quality   -- fraction of MATCHED objects that are crisp (not blurry); 1.0 if none matched
      precision -- matched / (matched + surplus-extra); how much of the scene is wanted; 1.0 if empty
    Substitution is already absorbed by match_enumeration (a wrong-color/wrong-shape object is paired to
    its expected slot), so `extra` is only genuinely surplus objects -- exactly what should cost precision.
    """
    objects, _, _ = parse_caption_gt(caption)
    n = len(objects)
    if n == 0:
        return None
    matches, extra = match_enumeration(objects, seen)
    n_extra = len(extra)
    credit = shape_ok = color_ok = present = blurry = 0.0
    for m in matches:
        if m["kind"] == "missing":
            continue
        present += 1
        s_ok = m["kind"] in ("exact", "color_bad")   # shape correct
        c_ok = m["kind"] in ("exact", "shape_bad")   # color correct
        shape_ok += s_ok
        color_ok += c_ok
        obj_credit = SHAPE_WEIGHT * s_ok + COLOR_WEIGHT * c_ok
        if m["blurry"]:
            obj_credit *= BLUR_KEEP
            blurry += 1
        credit += obj_credit
    return {
        "score": credit / (n + n_extra),   # surplus extras dilute the reward (precision cost)
        "presence": present / n,
        "shape": shape_ok / n,
        "color": color_ok / n,
        "quality": (1.0 - blurry / present) if present else 1.0,
        "precision": present / (present + n_extra) if (present + n_extra) else 1.0,
    }


def enumeration_feedback(caption, seen, max_edits=None):
    """Diff the enumerated scene against the caption entirely in code.

    Returns (command, rule, score). By default the command lists EVERY current error, ordered by the
    priority ladder (move -> replace/add missing -> shape -> color -> remove extra -> blurry last)
    and joined with "; ". The model regenerates the whole image at each feedback step, so telling it
    only one error invites fixing that one while breaking another; the full list states every
    constraint the regeneration has to satisfy. `max_edits=1` restores the old one-edit behaviour.
    """
    objects, _, _ = parse_caption_gt(caption)
    if not objects:
        return "no update", "match", None
    breakdown = enumeration_breakdown(caption, seen)
    score = breakdown["score"]
    matches, extra = match_enumeration(objects, seen)

    missing = [i for i, m in enumerate(matches) if m["kind"] == "missing"]
    wrong_shape = [i for i, m in enumerate(matches) if m["kind"] == "shape_bad"]  # right color, wrong shape
    wrong_color = [i for i, m in enumerate(matches) if m["kind"] == "color_bad"]  # right shape, wrong color
    extra = list(extra)

    def cs(i):
        return objects[i]["color"], objects[i]["shape"]

    def at(i):
        # only mention a cell when the caption actually specified one
        cell = objects[i].get("cell")
        return f" in cell {cell}" if cell is not None else ""

    edits = []          # (rule, command)

    # A misplaced object shows up as missing-at-its-cell plus an identical surplus elsewhere.
    # Name that directly instead of telling the model to add and remove the same thing.
    for i in list(missing):
        c, s_ = cs(i)
        hit = next((e for e in extra if e["color"] == c and e["shape"] == s_), None)
        if hit is None:
            continue
        src = hit.get("cell")
        where = f" from cell {src}" if src is not None else ""
        edits.append(("move", f"move the {c} {s_}{where} to cell {objects[i]['cell']}"
                              if objects[i].get("cell") is not None else f"move the {c} {s_}"))
        missing.remove(i); extra.remove(hit)

    while missing and extra:
        i, e = missing.pop(0), extra.pop(0)
        c, s_ = cs(i)
        edits.append(("extra-replace", f"replace the {e['color']} {e['shape']} with a {c} {s_}{at(i)}"))
    for i in missing:
        c, s_ = cs(i)
        edits.append(("missing", f"add a {c} {s_}{at(i)}"))
    for i in wrong_shape:
        c, s_ = cs(i)
        edits.append(("shape", f"fix shape of the {c} object{at(i)} to {s_}"))
    for i in wrong_color:
        c, s_ = cs(i)
        edits.append(("color", f"change color of the {matches[i]['seen']['color']} {s_}{at(i)} to {c}"))
    for e in extra:
        edits.append(("extra-remove", f"remove the {e['color']} {e['shape']}"))
    # blurry LAST: only once every expected object is present with correct color and shape
    if not edits:
        for i, m in enumerate(matches):
            if m["blurry"]:
                c, s_ = cs(i)
                edits.append(("blurry", f"fix the {c} {s_}{at(i)}"))

    if not edits:
        return "no update", "match", score
    if max_edits is not None:
        edits = edits[:int(max_edits)]
    return "; ".join(cmd for _, cmd in edits), edits[0][0], score


def total_tokens_from_usage(usage):
    for key in ("total_tokens", "totalTokenCount"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


class FeedbackVerifier:
    """Base verifier: subclasses implement the per-request _verify_row/_distance_row hooks
    (or override verify wholesale for natively batched backends); batches are threaded."""

    workers = 1

    def _map(self, fn, *columns):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(fn, *args) for args in zip(*columns)]
            return [future.result() for future in futures]

    def verify(self, captions, gt_images, attempt_images, feedback_histories=None):
        if feedback_histories is None:
            feedback_histories = [[] for _ in captions]
        assert len(captions) == len(gt_images) == len(attempt_images) == len(feedback_histories)
        return self._map(self._verify_row, captions, gt_images, attempt_images, feedback_histories)

    def score_distance(self, captions, gt_images, attempt_images):
        assert len(captions) == len(gt_images) == len(attempt_images)
        return self._map(self._distance_row, captions, gt_images, attempt_images)

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        raise NotImplementedError

    def _distance_row(self, caption, gt_image, attempt_image):
        raise NotImplementedError


class OpenAIChatVerifier(FeedbackVerifier):
    """Verifier over an OpenAI-compatible /chat/completions endpoint (OpenRouter, vLLM, ...)."""

    def __init__(
        self,
        api_key,
        model,
        api_url,
        temperature=0.0,
        max_tokens=32,
        retries=2,
        timeout=120,
        workers=8,
        enable_thinking=False,
        image_size=None,
        extra_payload=None,
        use_enumeration=False,
        top_p=None,
        top_k=None,
        presence_penalty=None,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = normalize_chat_url(api_url)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        self.timeout = int(timeout)
        self.workers = int(workers)
        self.enable_thinking = bool(enable_thinking)
        self.image_size = None if image_size is None else int(image_size)
        self.extra_payload = dict(extra_payload or {})
        # Enumeration mode: one caption-free "list what you see" call per image; the feedback command
        # AND the reward come from enumeration_feedback in code. See results/verifier_bench.
        self.use_enumeration = bool(use_enumeration)
        # Optional sampling params (vLLM supports them in the payload; None = server default).
        for key, value in (("top_p", top_p), ("top_k", top_k), ("presence_penalty", presence_penalty)):
            if value is not None:
                self.extra_payload[key] = value

    def _prepare_image(self, image):
        return resize_square(image, self.image_size) if self.image_size else image

    def _record_usage(self, usage):
        """Hook for subclasses to accumulate per-call usage stats (e.g. OpenRouter cost). No-op here."""

    def _enumeration_row(self, caption, attempt_image):
        """One enumeration call -> (command, score) via the code diff. The raw enumeration cannot fail
        to parse (invalid lines are dropped), so a transport-ok call always yields a usable result."""
        result = self._post_prompt(ENUMERATION_PROMPT, attempt_image, parse_feedback=False)
        if not result.ok:
            return result
        seen = parse_enumeration(result.feedback)
        command, _rule, score = enumeration_feedback(caption, seen)
        result.breakdown = enumeration_breakdown(caption, seen)
        result.feedback = command
        result.score = score
        return result

    def _verify_row(self, caption, gt_image, attempt_image, feedback_history):
        # gt_image is intentionally unused: feedback is judged against the caption only (no leakage).
        if self.use_enumeration:
            # Stateless: feedback_history is ignored -- the diff re-reports the worst remaining error,
            # which changes as the image improves, so repeats only happen while the error persists.
            return self._enumeration_row(caption, attempt_image)
        return self._post_prompt(
            build_feedback_prompt(
                caption,
                feedback_history=feedback_history,
                enable_thinking=self.enable_thinking,
            ),
            attempt_image,
            parse_feedback=True,
        )

    def _distance_row(self, caption, gt_image, attempt_image):
        # gt_image unused: the structured check is judged against the caption, not the GT image.
        if self.use_enumeration:
            return self._enumeration_row(caption, attempt_image)
        result = self._post_prompt(build_distance_prompt(caption), attempt_image, parse_feedback=False)
        if not result.ok:
            return result
        score = score_structured_distance(result.feedback, caption)
        if score is None:
            return VerificationResult(
                ok=False, error="unparseable distance JSON",
                token_count=result.token_count, token_usage=result.token_usage,
            )
        result.score = score
        return result

    def _post_prompt(self, prompt, attempt_image, parse_feedback=True):
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Generated image:"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(self._prepare_image(attempt_image))}},
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_payload)
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                usage = result.get("usage", {}) or {}
                self._record_usage(usage)
                text = result["choices"][0]["message"]["content"]
                feedback = clean_feedback_text(text) if parse_feedback else str(text).strip()
                if not feedback and parse_feedback:
                    raise ValueError("empty feedback")
                return VerificationResult(
                    ok=True,
                    feedback=feedback,
                    score=None,
                    token_count=total_tokens_from_usage(usage),
                    token_usage=usage,
                )
            except Exception as exc:
                if attempt >= self.retries:
                    return VerificationResult(ok=False, error=repr(exc))
                time.sleep(2 ** attempt)
        return VerificationResult(ok=False, error="unreachable")
