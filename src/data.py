"""Read the existing index without modifying the dataset."""
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


LABELS = {"BlackSigatoka": 0, "FusariumWilt": 1, "Healthy": 2}
TEMPLATES = (
    "a photo of a banana sample with black sigatoka",
    "a photo of a banana sample with fusarium wilt",
    "a photo of a healthy banana sample",
)
MODES = {"none", "coarse", "coarse_caption", "caption"}


class BananaDataset(Dataset):
    def __init__(self, root, split, text_mode):
        self.root = Path(root).resolve()
        self.split = {"val": "validation"}.get(split, split)
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if text_mode not in MODES:
            raise ValueError(f"Invalid text_mode: {text_mode}")
        self.text_mode = text_mode
        self.records = []
        self.by_label = {i: [] for i in range(3)}
        source = self.root / "datasets" / "banana" / "index.jsonl"
        with source.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                item = json.loads(line)
                if item["split"] != self.split:
                    continue
                raw_labels = item["class_labels"]
                if len(raw_labels) != 1 or raw_labels[0] not in LABELS:
                    raise ValueError(f"Unsupported banana label at line {line_number}")
                label = LABELS[raw_labels[0]]
                image_path = (self.root / item["image_path"]).resolve()
                if not image_path.is_relative_to(self.root) or not image_path.is_file():
                    raise ValueError(f"Invalid image path at line {line_number}")
                caption = item["blind_caption"]
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"Empty caption at line {line_number}")
                record = {"image_id": item["image_id"], "path": image_path,
                          "label": label, "caption": caption,
                          "split_group_id": item.get("split_group_id", item["image_id"])}
                self.by_label[label].append(len(self.records))
                self.records.append(record)
        if not self.records or any(not entries for entries in self.by_label.values()):
            raise ValueError(f"Empty split or missing class: {self.split}")
        ids = [r["image_id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate image_id in split {self.split}")

    def text(self, index):
        r = self.records[index]
        if self.text_mode == "none":
            return ""
        if self.text_mode == "coarse":
            return TEMPLATES[r["label"]]
        if self.text_mode == "coarse_caption":
            return TEMPLATES[r["label"]] + ". " + r["caption"]
        return r["caption"]

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(r["path"]) as image:
            image = image.convert("RGB")
        return {"image": image, "image_id": r["image_id"], "label": r["label"],
                "text": self.text(index)}

    def __len__(self):
        return len(self.records)


def verify_disjoint(datasets):
    for i, left in enumerate(datasets):
        left_ids = {r["image_id"] for r in left.records}
        left_groups = {r["split_group_id"] for r in left.records}
        for right in datasets[i + 1:]:
            if left_ids.intersection(r["image_id"] for r in right.records):
                raise ValueError(f"image IDs overlap: {left.split}, {right.split}")
            if left_groups.intersection(r["split_group_id"] for r in right.records):
                raise ValueError(f"split groups overlap: {left.split}, {right.split}")


def collate(rows):
    return {k: [row[k] for row in rows] for k in ("image", "image_id", "label", "text")}
