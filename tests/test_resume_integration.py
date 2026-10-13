"""Real small CPU trainings: partial best.pt must never count as completed."""
import csv
import json

import pytest
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.patches import torch_load

from plantdoc_tcc.checkpoints import checkpoint_state
from plantdoc_tcc.staged import _reconcile_summary, load_protocol, loss_specs, run_phase


@pytest.mark.parametrize("strategy", ["baseline", "class_weighted", "focal"])
def test_resume_after_saved_epoch(tmp_path, monkeypatch, strategy):
    data = tmp_path / "data"
    for split in ("train", "val"):
        (data / split / "images").mkdir(parents=True)
        (data / split / "labels").mkdir()
        for index in range(4):
            Image.new("RGB", (64, 64), (20, 100, 20)).save(data / split / "images" / f"{index}.jpg")
            (data / split / "labels" / f"{index}.txt").write_text(
                f"{index % 2} 0.5 0.5 0.5 0.5\n"
            )
    yaml = data / "data.yaml"
    yaml.write_text(f"path: {data.as_posix()}\ntrain: train/images\nval: val/images\nnames: [a, b]\n")
    protocol = load_protocol("configs/staged_smoke.json")
    protocol["patience"] = 0
    protocol["loss_comparison"].update(epochs=3, strategies=[strategy])
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(protocol))
    project = tmp_path / "runs"
    original_train = YOLO.train
    interrupted = False
    resumed_epochs = []

    def interrupt(trainer):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated session interruption")

    def wrapped_train(self, *args, **kwargs):
        if not interrupted:
            self.add_callback("on_model_save", interrupt)
        else:
            self.add_callback("on_train_epoch_start", lambda t: resumed_epochs.append(t.epoch))
        kwargs.update(plots=False, amp=False)
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(YOLO, "train", wrapped_train)
    with pytest.raises(KeyboardInterrupt):
        run_phase(str(yaml), str(config), str(project), "loss", "cpu", 1)
    directory = project / "01_loss" / loss_specs(protocol)[0].name
    assert (directory / "weights" / "best.pt").exists()
    assert checkpoint_state(directory, 3) == {"state": "interrupted", "saved_epoch": 1}
    assert not (project / "protocol_summary.json").exists()
    ckpt = torch_load(directory / "weights" / "last.pt", map_location="cpu")
    assert ckpt["optimizer"] is not None
    if strategy != "baseline":
        assert ckpt["ema"].loss_class_weights is not None
    summary = run_phase(str(yaml), str(config), str(project), "loss", "cpu", 1)
    assert resumed_epochs == [1, 2]
    backup = torch_load(directory / "weights" / "last.before_resume.pt", map_location="cpu")
    assert backup["epoch"] == 0 and backup["optimizer"] is not None
    assert checkpoint_state(directory, 3)["state"] == "completed"
    assert summary["winning_loss"] == strategy
    with (directory / "results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    # Rerunning a finished item must not invoke training again.
    monkeypatch.setattr(YOLO, "train", lambda *a, **k: pytest.fail("retrained completed run"))
    run_phase(str(yaml), str(config), str(project), "loss", "cpu", 1)


def test_stale_completion_is_removed_without_deleting_artifacts(tmp_path):
    import torch

    run = tmp_path / "01_loss" / "partial"
    (run / "weights").mkdir(parents=True)
    torch.save({"epoch": 28, "optimizer": {"state": {}}, "train_args": {"epochs": 100}},
               run / "weights" / "last.pt")
    (run / "weights" / "best.pt").touch()
    (run / "results.csv").write_text("epoch\n29\n")
    summary = {"runs": {"loss": [{"name": "partial", "epochs": 100}],
                        "search": [{"name": "old", "epochs": 100}]},
               "winning_loss": "focal", "best_model": {"name": "old"}}
    _reconcile_summary(tmp_path, summary)
    assert summary["runs"]["loss"] == [] and summary["runs"]["search"] == []
    assert "winning_loss" not in summary and "best_model" not in summary
    assert checkpoint_state(run, 100) == {"state": "interrupted", "saved_epoch": 29}


def test_corrupt_checkpoint_is_not_treated_as_completed(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "last.pt").write_bytes(b"incomplete file")
    (tmp_path / "weights" / "best.pt").touch()
    (tmp_path / "results.csv").touch()
    assert checkpoint_state(tmp_path, 100)["state"] == "unreadable"
