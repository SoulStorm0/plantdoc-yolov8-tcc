from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetAudit:
    nc: int
    names: list[str]
    image_count: int
    labeled_image_count: int
    empty_image_count: int
    invalid_lines: int
    instance_counts: list[int]


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Instale PyYAML para ler data.yaml: pip install -e .") from exc
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("data.yaml deve conter um mapeamento.")
    return value


def normalize_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        return [str(value[k]) for k in sorted(value, key=lambda x: int(x))]
    raise ValueError("O campo names deve ser lista ou dicionário indexado.")


def resolve_split(data_yaml: str | Path, split: str) -> Path:
    yaml_path = Path(data_yaml).resolve()
    data = load_yaml(yaml_path)
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    raw = data.get(split)
    if isinstance(raw, list) or raw is None:
        raise ValueError(f"Este comando requer que '{split}' seja um único diretório.")
    split_path = Path(raw)
    return split_path if split_path.is_absolute() else (root / split_path).resolve()


def image_to_label_path(image: Path) -> Path:
    parts = list(image.parts)
    indices = [i for i, item in enumerate(parts) if item.lower() == "images"]
    if not indices:
        raise ValueError(f"Caminho sem diretório images: {image}")
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def audit_dataset(data_yaml: str | Path, split: str = "train", expected_classes: int | None = None):
    data = load_yaml(data_yaml)
    names = normalize_names(data["names"])
    nc = int(data.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"nc={nc}, mas names possui {len(names)} entradas.")
    if expected_classes is not None and nc != expected_classes:
        raise ValueError(f"Esperadas {expected_classes} classes; o dataset declara {nc}.")

    image_dir = resolve_split(data_yaml, split)
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    counts: Counter[int] = Counter()
    labeled = invalid = empty = 0
    for image in images:
        label = image_to_label_path(image)
        if not label.exists() or not label.read_text(encoding="utf-8").strip():
            empty += 1
            continue
        labeled += 1
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            try:
                class_id = int(fields[0])
                coords = [float(x) for x in fields[1:]]
                valid = len(coords) == 4 and 0 <= class_id < nc and all(0 <= x <= 1 for x in coords)
                if not valid or coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError
                counts[class_id] += 1
            except (ValueError, IndexError):
                invalid += 1
    return DatasetAudit(nc, names, len(images), labeled, empty, invalid, [counts[i] for i in range(nc)])


def class_weights(counts: list[int], method: str = "sqrt_inverse_frequency", max_weight: float = 8.0):
    if not counts or any(x < 0 for x in counts):
        raise ValueError("Contagens inválidas.")
    if any(x == 0 for x in counts):
        missing = [i for i, x in enumerate(counts) if x == 0]
        raise ValueError(f"Classes sem instâncias no treino: {missing}")
    maximum = max(counts)
    if method == "inverse_frequency":
        raw = [maximum / x for x in counts]
    elif method == "sqrt_inverse_frequency":
        raw = [math.sqrt(maximum / x) for x in counts]
    else:
        raise ValueError(f"Método de ponderação desconhecido: {method}")
    clipped = [min(max_weight, x) for x in raw]
    mean = sum(clipped) / len(clipped)
    return [x / mean for x in clipped]

