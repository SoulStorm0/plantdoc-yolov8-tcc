"""Converte o repositório oficial PlantDoc (Pascal VOC) para YOLO.

O repositório contém nomes incompatíveis com Windows (por exemplo, `?` e `&`).
Por isso os blobs são lidos diretamente do Git e gravados com nomes SHA-256 seguros.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


@dataclass
class Sample:
    source_image: str
    image_bytes: bytes
    width: int
    height: int
    objects: list[tuple[str, float, float, float, float]]

    @property
    def classes(self) -> set[str]:
        return {item[0] for item in self.objects}


class GitBlobReader:
    def __init__(self, repo: Path):
        self.process = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def read(self, revision_path: str) -> bytes:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(f"HEAD:{revision_path}\n".encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().decode().strip().split()
        if len(header) != 3 or header[1] != "blob":
            raise ValueError(f"Blob Git inválido: {revision_path}: {header}")
        payload = self.process.stdout.read(int(header[2]))
        self.process.stdout.read(1)
        return payload

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=10)


def git_paths(repo: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "-z", "HEAD"]
    )
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def parse_annotation(xml_bytes: bytes) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    root = ET.fromstring(xml_bytes)
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError("Anotação sem dimensões válidas.")
    objects = []
    for node in root.findall("object"):
        name = (node.findtext("name") or "").strip()
        box = node.find("bndbox")
        if not name or box is None:
            continue
        xmin = max(0.0, float(box.findtext("xmin", "0")))
        ymin = max(0.0, float(box.findtext("ymin", "0")))
        xmax = min(float(width), float(box.findtext("xmax", "0")))
        ymax = min(float(height), float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append(
            (
                name,
                ((xmin + xmax) / 2) / width,
                ((ymin + ymax) / 2) / height,
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
        )
    return width, height, objects


def match_pairs(paths: list[str]) -> list[tuple[str, str]]:
    images = {}
    annotations = {}
    for path in paths:
        pure = PurePosixPath(path)
        if pure.parts[0] not in {"TRAIN", "TEST"}:
            continue
        key = (str(pure.parent), pure.stem.casefold())
        if pure.suffix.casefold() == ".xml":
            annotations[key] = path
        elif pure.suffix.casefold() in IMAGE_SUFFIXES:
            images[key] = path
    return [(images[key], annotations[key]) for key in sorted(images.keys() & annotations.keys())]


def assign_splits(samples: list[Sample], seed: int = 42) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    class_totals = Counter(name for sample in samples for name in sample.classes)
    targets = {
        split: {name: total * ratio for name, total in class_totals.items()}
        for split, ratio in SPLIT_RATIOS.items()
    }
    size_targets = {split: len(samples) * ratio for split, ratio in SPLIT_RATIOS.items()}
    assigned = {split: [] for split in SPLIT_RATIOS}
    counts = {split: Counter() for split in SPLIT_RATIOS}
    samples.sort(key=lambda sample: min(class_totals[x] for x in sample.classes) if sample.classes else 10**9)
    for sample in samples:
        def score(split: str):
            class_need = sum(max(0.0, targets[split][name] - counts[split][name]) for name in sample.classes)
            size_need = max(0.0, size_targets[split] - len(assigned[split]))
            overflow = max(0.0, len(assigned[split]) + 1 - round(size_targets[split]))
            return class_need + size_need / max(1, len(samples)) - overflow * 1000

        destination = max(SPLIT_RATIOS, key=score)
        assigned[destination].append(sample)
        counts[destination].update(sample.classes)
    return assigned


def safe_name(source: str) -> str:
    suffix = PurePosixPath(source).suffix.casefold()
    return hashlib.sha256(source.encode()).hexdigest()[:24] + suffix


def convert(repo: Path, output: Path, seed: int, min_class_instances: int):
    paths = git_paths(repo)
    pairs = match_pairs(paths)
    reader = GitBlobReader(repo)
    samples = []
    failures = []
    try:
        for image_path, xml_path in pairs:
            try:
                width, height, objects = parse_annotation(reader.read(xml_path))
                samples.append(Sample(image_path, reader.read(image_path), width, height, objects))
            except Exception as exc:  # Registra arquivos problemáticos sem ocultar o total.
                failures.append((xml_path, str(exc)))
    finally:
        reader.close()

    raw_counts = Counter(name for sample in samples for name, *_ in sample.objects)
    excluded = {name for name, total in raw_counts.items() if total < min_class_instances}
    # Descarta a imagem completa quando contém classe excluída. Remover somente a caixa
    # transformaria o objeto visível em falso background e contaminaria o treinamento.
    samples = [sample for sample in samples if not (sample.classes & excluded)]
    names = sorted({name for sample in samples for name in sample.classes})
    class_ids = {name: index for index, name in enumerate(names)}
    splits = assign_splits(samples, seed)
    for split, entries in splits.items():
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for sample in entries:
            filename = safe_name(sample.source_image)
            (image_dir / filename).write_bytes(sample.image_bytes)
            lines = [
                f"{class_ids[name]} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"
                for name, x, y, w, h in sample.objects
            ]
            (label_dir / Path(filename).with_suffix(".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    quoted_names = ", ".join(repr(name) for name in names)
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames: [{quoted_names}]\n",
        encoding="utf-8",
    )
    report = [
        f"paired_samples={len(samples)}",
        f"classes={len(names)}",
        f"minimum_class_instances={min_class_instances}",
        "excluded_classes=" + (" | ".join(sorted(excluded)) if excluded else "none"),
        *(f"{split}={len(entries)}" for split, entries in splits.items()),
        f"failures={len(failures)}",
        "class_names=" + " | ".join(names),
    ]
    if failures:
        report.extend(f"failure={path}: {error}" for path, error in failures)
    (output / "conversion_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("datasets/plantdoc_official"))
    parser.add_argument("--output", type=Path, default=Path("datasets/plantdoc_yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-class-instances", type=int, default=20)
    args = parser.parse_args()
    convert(args.repo.resolve(), args.output.resolve(), args.seed, args.min_class_instances)


if __name__ == "__main__":
    main()
