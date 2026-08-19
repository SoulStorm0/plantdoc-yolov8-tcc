from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import iter_specs
from .data import audit_dataset, class_weights


def train_grid(
    data_yaml: str,
    config: dict,
    strategies: Iterable[str] | None = None,
    device=None,
    epochs: Iterable[int] | None = None,
):
    from ultralytics import YOLO

    audit = audit_dataset(data_yaml, "train")
    weighting = config.get("class_weighting", {})
    weights = class_weights(
        audit.instance_counts,
        weighting.get("method", "sqrt_inverse_frequency"),
        float(weighting.get("max_weight", 8.0)),
    )
    project = Path(config.get("project", "runs/plantdoc")).resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / "dataset_audit.json").write_text(
        json.dumps({**audit.__dict__, "class_weights": weights}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    from .losses import make_trainer_class

    selected_epochs = None if epochs is None else set(epochs)
    for spec in iter_specs(config, strategies):
        if selected_epochs is not None and spec.epochs not in selected_epochs:
            continue
        trainer = None
        if spec.loss_strategy == "class_weighted":
            trainer = make_trainer_class(weights=weights)
        elif spec.loss_strategy == "focal":
            focal = config.get("focal", {})
            trainer = make_trainer_class(
                weights=weights,
                focal_gamma=float(focal.get("gamma", 2.0)),
                focal_alpha=float(focal.get("alpha", 0.25)),
            )
        model = YOLO(config.get("model", "yolov8n.pt"))
        augmentation = config.get("augmentation", {})
        arguments = dict(
            data=str(Path(data_yaml).resolve()),
            project=str(project),
            name=spec.name,
            exist_ok=False,
            epochs=spec.epochs,
            batch=spec.batch,
            imgsz=int(config.get("imgsz", 640)),
            optimizer=config.get("optimizer", "SGD"),
            lr0=spec.lr0,
            lrf=float(config.get("lrf", 0.01)),
            momentum=spec.momentum,
            weight_decay=spec.weight_decay,
            warmup_epochs=float(config.get("warmup_epochs", 5.0)),
            warmup_momentum=float(config.get("warmup_momentum", 0.8)),
            cos_lr=bool(config.get("cos_lr", True)),
            patience=int(config.get("patience", 30)),
            seed=int(config.get("seed", 42)),
            deterministic=bool(config.get("deterministic", True)),
            fraction=float(config.get("fraction", 1.0)),
            workers=int(config.get("workers", 8)),
            device=device,
            **augmentation,
        )
        if trainer is not None:
            arguments["trainer"] = trainer
        model.train(**arguments)
