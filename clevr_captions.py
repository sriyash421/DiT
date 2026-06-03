"""
Caption rendering utilities for structured CLEVR metadata.
"""


def full_description(obj):
    return f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}"


def label_candidates(obj):
    return [
        f"{obj['color']} {obj['shape']}",
        f"{obj['size']} {obj['color']} {obj['shape']}",
        f"{obj['color']} {obj['material']} {obj['shape']}",
        full_description(obj),
    ]


def unique_labels(objects):
    labels = [None] * len(objects)
    for level in range(4):
        candidates = [label_candidates(obj)[level] for obj in objects]
        counts = {candidate: candidates.count(candidate) for candidate in candidates}
        for idx, candidate in enumerate(candidates):
            if labels[idx] is None and counts[candidate] == 1:
                labels[idx] = candidate
    for idx, label in enumerate(labels):
        if label is None:
            labels[idx] = f"object {idx + 1} {full_description(objects[idx])}"
    return labels


def adjacent_chain(labels, order, relation):
    return ", ".join(
        f"{labels[order[idx + 1]]} is {relation} {labels[order[idx]]}"
        for idx in range(len(order) - 1)
    )


def render_caption(row, template="chain"):
    objects = row["objects"]
    labels = unique_labels(objects)
    object_list = ", ".join(full_description(obj) for obj in objects)
    left_to_right = row["orders"]["left_to_right"]
    front_to_back = row["orders"]["front_to_back"]

    if template == "chain":
        horizontal = adjacent_chain(labels, left_to_right, "right of")
        depth = adjacent_chain(labels, front_to_back, "behind")
        return f"objects: {object_list}. horizontal: {horizontal}. depth: {depth}."

    if template == "order":
        horizontal = ", ".join(labels[idx] for idx in left_to_right)
        depth = ", ".join(labels[idx] for idx in front_to_back)
        return f"objects: {object_list}. left-to-right: {horizontal}. front-to-back: {depth}."

    if template == "compact":
        horizontal = adjacent_chain(labels, left_to_right, "right of")
        depth = adjacent_chain(labels, front_to_back, "behind")
        return f"{object_list}. {horizontal}. {depth}."

    raise ValueError(f"Unknown CLEVR caption template: {template}")
