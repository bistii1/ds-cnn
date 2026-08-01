"""Load and expose the pipeline configuration from config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Standard directory layout for the pipeline.
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"   # raw corpora as downloaded
RAW_DIR = PROJECT_ROOT / "raw"               # per-label wavs extracted from corpora
PROCESSED_DIR = PROJECT_ROOT / "processed"   # normalized + padded wavs, ready to upload


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def all_target_labels(cfg: dict[str, Any]) -> dict[str, str]:
    """Return {label: source_corpus} for every target word across all sources."""
    labels: dict[str, str] = {}
    for source, mapping in cfg["vocabulary"].items():
        for label in mapping:
            labels[label] = source
    return labels


if __name__ == "__main__":
    cfg = load_config()
    labels = all_target_labels(cfg)
    print(f"{len(labels)} classes (real CC-BY only, no TTS)")
    print(f"max_per_class: {cfg['sampling']['max_per_class']}")
    print(f"balance_target: {cfg['sampling']['balance_target']} clips per label after make balance")
    for label, source in labels.items():
        print(f"  {label:14s} <- {source}")
    print(f"\nTotal classes: {len(labels)}")
