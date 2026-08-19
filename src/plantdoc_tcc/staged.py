from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import RunSpec
from .data import audit_dataset


METRIC_MAP = "metrics/mAP50-95(B)"
METRIC_MAP50 = "metrics/mAP50(B)"
PHASES = ("loss", "search", "promote200", "confirm300")


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    required = {"loss_comparison", "search", "promotion", "confirmation"}
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"Protocolo sem seções obrigatórias: {sorted(missing)}")
    return protocol


def loss_specs(protocol: dict[str, Any]) -> list[RunSpec]:
    fixed = protocol["loss_comparison"]
    return [
        RunSpec(
            int(fixed["epochs"]),
            int(fixed["batch"]),
            float(fixed["lr0"]),
            float(fixed["momentum"]),
            float(fixed["weight_decay"]),
            strategy,
        )
        for strategy in fixed["strategies"]
    ]


def search_specs(protocol: dict[str, Any], strategy: str) -> list[RunSpec]:
    search = protocol["search"]
    return [
        RunSpec(
            int(search["epochs"]),
            int(trial["batch"]),
            float(trial["lr0"]),
            float(trial["momentum"]),
            float(trial["weight_decay"]),
            strategy,
        )
        for trial in search["trials"]
    ]


def score_results(path: str | Path) -> tuple[float, float]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Arquivo de resultados vazio: {path}")
    best = max(rows, key=lambda row: (float(row[METRIC_MAP]), float(row[METRIC_MAP50])))
    return float(best[METRIC_MAP]), float(best[METRIC_MAP50])


def _base_config(protocol: dict[str, Any], project: Path, spec: RunSpec) -> dict[str, Any]:
    return {
        "project": str(project),
        "model": protocol.get("model", "yolov8n.pt"),
        "imgsz": int(protocol.get("imgsz", 640)),
        "seed": int(protocol.get("seed", 42)),
        "deterministic": bool(protocol.get("deterministic", True)),
        "exist_ok": True,
        "optimizer": protocol.get("optimizer", "SGD"),
        "epochs": [spec.epochs],
        "batch": [spec.batch],
        "lr0": [spec.lr0],
        "momentum": [spec.momentum],
        "weight_decay": [spec.weight_decay],
        "loss_strategies": [spec.loss_strategy],
        "lrf": float(protocol.get("lrf", 0.01)),
        "warmup_epochs": float(protocol.get("warmup_epochs", 5.0)),
        "warmup_momentum": float(protocol.get("warmup_momentum", 0.8)),
        "cos_lr": bool(protocol.get("cos_lr", True)),
        "patience": int(protocol.get("patience", 30)),
        "workers": int(protocol.get("workers", 2)),
        "class_weighting": protocol.get("class_weighting", {}),
        "focal": protocol.get("focal", {}),
        "augmentation": protocol.get("augmentation", {}),
    }


def _run_spec(data: str, protocol: dict[str, Any], phase_dir: Path, spec: RunSpec, device):
    run_dir = phase_dir / spec.name
    weights = run_dir / "weights" / "best.pt"
    results = run_dir / "results.csv"
    if not (weights.exists() and results.exists()):
        from .runner import train_grid

        train_grid(data, _base_config(protocol, phase_dir, spec), [spec.loss_strategy], device, [spec.epochs])
    map_value, map50 = score_results(results)
    return {
        **asdict(spec),
        "name": spec.name,
        "phase": phase_dir.name,
        "map50_95": map_value,
        "map50": map50,
        "weights": str(weights.resolve()),
        "results_csv": str(results.resolve()),
    }


def _save_summary(project: Path, summary: dict[str, Any]):
    project.mkdir(parents=True, exist_ok=True)
    (project / "protocol_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [row for phase in PHASES for row in summary.get("runs", {}).get(phase, [])]
    if rows:
        with (project / "protocol_runs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_phase(
    data: str,
    protocol_path: str,
    project_path: str,
    phase: str,
    device: str | int | None = 0,
) -> dict[str, Any]:
    if phase not in (*PHASES, "all"):
        raise ValueError(f"Fase inválida: {phase}")
    protocol = load_protocol(protocol_path)
    audit = audit_dataset(
        data, "train", expected_classes=int(protocol.get("expected_classes", 27))
    )
    if audit.invalid_lines:
        raise ValueError("Dataset contém rótulos YOLO inválidos.")
    project = Path(project_path).resolve()
    summary_path = project / "protocol_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {"data": str(Path(data).resolve()), "runs": {}, "final_test_completed": False}
    )

    requested = PHASES if phase == "all" else (phase,)
    for current in requested:
        if current == "loss":
            runs = [_run_spec(data, protocol, project / "01_loss", spec, device) for spec in loss_specs(protocol)]
            summary["runs"][current] = runs
            winner = max(runs, key=lambda row: (row["map50_95"], row["map50"]))
            summary["winning_loss"] = winner["loss_strategy"]
        elif current == "search":
            strategy = summary.get("winning_loss")
            if not strategy:
                raise RuntimeError("Execute primeiro a fase loss.")
            runs = [
                _run_spec(data, protocol, project / "02_search", spec, device)
                for spec in search_specs(protocol, strategy)
            ]
            summary["runs"][current] = runs
        elif current == "promote200":
            candidates = summary.get("runs", {}).get("search")
            if not candidates:
                raise RuntimeError("Execute primeiro a fase search.")
            top_k = int(protocol["promotion"]["top_k"])
            selected = sorted(candidates, key=lambda row: (row["map50_95"], row["map50"]), reverse=True)[:top_k]
            specs = [
                RunSpec(
                    int(protocol["promotion"]["epochs"]),
                    int(row["batch"]),
                    float(row["lr0"]),
                    float(row["momentum"]),
                    float(row["weight_decay"]),
                    row["loss_strategy"],
                )
                for row in selected
            ]
            summary["runs"][current] = [
                _run_spec(data, protocol, project / "03_promote200", spec, device) for spec in specs
            ]
        elif current == "confirm300":
            candidates = summary.get("runs", {}).get("promote200")
            if not candidates:
                raise RuntimeError("Execute primeiro a fase promote200.")
            winner = max(candidates, key=lambda row: (row["map50_95"], row["map50"]))
            spec = RunSpec(
                int(protocol["confirmation"]["epochs"]),
                int(winner["batch"]),
                float(winner["lr0"]),
                float(winner["momentum"]),
                float(winner["weight_decay"]),
                winner["loss_strategy"],
            )
            run = _run_spec(data, protocol, project / "04_confirm300", spec, device)
            summary["runs"][current] = [run]
            summary["best_model"] = run
        _save_summary(project, summary)
    return summary


def finalize_test(data: str, project_path: str) -> dict[str, Any]:
    project = Path(project_path).resolve()
    summary_path = project / "protocol_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Resumo não encontrado; execute o protocolo até confirm300.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best_model")
    if not best:
        raise RuntimeError("Melhor modelo ainda não foi confirmado em 300 épocas.")
    output = project / "final_test_metrics.json"
    if summary.get("final_test_completed") or output.exists():
        raise RuntimeError("O conjunto de teste final já foi avaliado; repetição bloqueada.")
    from .evaluate import evaluate

    metrics = evaluate(best["weights"], data, "test", str(output))
    summary["final_test_completed"] = True
    summary["final_test_metrics"] = str(output)
    _save_summary(project, summary)
    return metrics
