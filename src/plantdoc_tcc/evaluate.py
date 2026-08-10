from __future__ import annotations

import csv
import json
from pathlib import Path


def metrics_payload(metrics) -> dict:
    box = metrics.box
    names = metrics.names
    indices = [int(x) for x in box.ap_class_index]
    classes = []
    for position, class_id in enumerate(indices):
        classes.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "precision": float(box.p[position]),
                "recall": float(box.r[position]),
                "f1": float(box.f1[position]),
                "ap50": float(box.ap50[position]),
                "ap50_95": float(box.ap[position]),
            }
        )
    return {
        "global": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": classes,
    }


def evaluate(weights: str, data: str, split: str = "test", output: str = "artifacts/metrics.json"):
    from ultralytics import YOLO

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    run_name = f"{split}_{Path(weights).resolve().parent.parent.name}"
    metrics = YOLO(str(Path(weights).resolve())).val(
        data=str(Path(data).resolve()),
        split=split,
        plots=True,
        project=str((target.parent / "evaluation").resolve()),
        name=run_name,
        exist_ok=True,
    )
    payload = metrics_payload(metrics)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def validate_external_metadata(metadata_csv: str | Path, image_root: str | Path):
    required = {"image", "lighting", "annotator_1", "annotator_2", "adjudicated"}
    rows = []
    with Path(metadata_csv).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata.csv sem colunas: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            image = (Path(image_root) / row["image"]).resolve()
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            if not image.exists() or not label.exists():
                raise ValueError(f"Linha {line}: imagem ou ground truth ausente.")
            if row["lighting"] not in {"baixa", "difusa", "direta", "contraluz"}:
                raise ValueError(f"Linha {line}: condição de iluminação inválida.")
            if row["adjudicated"].lower() not in {"true", "1", "sim"}:
                raise ValueError(f"Linha {line}: anotação ainda não adjudicada.")
            rows.append(row)
    return rows
