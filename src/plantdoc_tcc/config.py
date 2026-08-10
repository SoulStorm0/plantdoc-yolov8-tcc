from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunSpec:
    epochs: int
    batch: int
    lr0: float
    momentum: float
    weight_decay: float
    loss_strategy: str

    @property
    def name(self) -> str:
        lr = f"{self.lr0:.0e}".replace("-", "m")
        wd = f"{self.weight_decay:.0e}".replace("-", "m")
        return (
            f"{self.loss_strategy}_e{self.epochs}_b{self.batch}_lr{lr}_"
            f"m{self.momentum:g}_wd{wd}"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"epochs", "batch", "lr0", "momentum", "weight_decay", "loss_strategies"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Configuração sem campos obrigatórios: {sorted(missing)}")
    if any(not 1e-5 <= float(x) <= 1e-3 for x in config["lr0"]):
        raise ValueError("O grid deste protocolo exige lr0 entre 1e-5 e 1e-3.")
    allowed = {"baseline", "class_weighted", "focal"}
    unknown = set(config["loss_strategies"]) - allowed
    if unknown:
        raise ValueError(f"Estratégias de loss desconhecidas: {sorted(unknown)}")


def iter_specs(config: dict[str, Any], strategies: Iterable[str] | None = None):
    selected = list(strategies or config["loss_strategies"])
    for values in itertools.product(
        config["epochs"],
        config["batch"],
        config["lr0"],
        config["momentum"],
        config["weight_decay"],
        selected,
    ):
        yield RunSpec(*values)

