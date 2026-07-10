"""Dataset factories used by the hydra configs."""
from datasets.clevr.dataset import ClevrContextMultiDataset
from datasets.clevr.utils import build_clevr_transform


def build_clevr_dataset(
    datasets,
    split,
    image_size,
    use_disk=True,
    load_meta=True,
    load_images=False,
    load_context=False,
):
    """Build a weighted multi-source CLEVR dataset from config entries (name, path, sampling_ratio, max_dataset_size)."""
    entries = [
        {
            "name": entry["name"],
            "path": entry["path"],
            "sampling_ratio": float(entry["sampling_ratio"]),
            "max_dataset_size": entry["max_dataset_size"],
        }
        for entry in datasets
    ]
    return ClevrContextMultiDataset(
        entries,
        transform=build_clevr_transform(image_size),
        split=split,
        use_disk=use_disk,
        load_meta=load_meta,
        load_images=load_images,
        load_context=load_context,
    )
