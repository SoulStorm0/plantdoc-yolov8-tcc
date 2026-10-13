"""Inspect only checkpoints produced by this project (pickle files must be trusted)."""
from pathlib import Path


def checkpoint_state(run_dir: Path, expected_epochs: int) -> dict:
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    results = run_dir / "results.csv"
    if not last.exists():
        state = "missing_checkpoint" if best.exists() or results.exists() else "pending"
        return {"state": state}
    try:
        from ultralytics.utils.patches import torch_load

        ckpt = torch_load(str(last), map_location="cpu")
        epoch = ckpt.get("epoch")
        target = ckpt.get("train_args", {}).get("epochs")
        if target != expected_epochs:
            return {"state": "incompatible", "checkpoint_target": target}
        if epoch == -1 and ckpt.get("optimizer") is None and best.exists() and results.exists():
            # Ultralytics strips optimizer and sets epoch=-1 only at final evaluation,
            # including legitimate early stopping. A partial best.pt is not proof.
            return {"state": "completed"}
        if isinstance(epoch, int) and epoch >= 0 and ckpt.get("optimizer") is not None:
            return {"state": "interrupted", "saved_epoch": epoch + 1}
        return {"state": "unverified"}
    except Exception as exc:
        return {"state": "unreadable", "error": str(exc)}
