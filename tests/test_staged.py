import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from plantdoc_tcc.staged import (
    _merge_runs,
    _select_specs,
    load_protocol,
    loss_specs,
    protocol_status,
    run_phase,
    score_results,
    search_specs,
)


class StagedProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol("configs/colab_protocol.json")

    def test_protocol_has_expected_staged_workload(self):
        self.assertEqual(len(loss_specs(self.protocol)), 3)
        self.assertEqual(len(search_specs(self.protocol, "focal")), 8)
        values = {spec.lr0 for spec in search_specs(self.protocol, "focal")}
        self.assertEqual(values, {1e-3, 1e-4, 1e-5})

    def test_score_uses_best_validation_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["metrics/mAP50-95(B)", "metrics/mAP50(B)"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"metrics/mAP50-95(B)": "0.2", "metrics/mAP50(B)": "0.5"},
                        {"metrics/mAP50-95(B)": "0.4", "metrics/mAP50(B)": "0.6"},
                    ]
                )
            self.assertEqual(score_results(path), (0.4, 0.6))

    def test_protocol_is_json_serializable(self):
        json.dumps(self.protocol)

    def test_local_check_is_small_and_keeps_all_loss_strategies(self):
        protocol = load_protocol("configs/local_check.json")
        self.assertEqual(protocol["fraction"], 0.05)
        self.assertEqual(protocol["imgsz"], 320)
        self.assertEqual([spec.loss_strategy for spec in loss_specs(protocol)], [
            "baseline", "class_weighted", "focal"
        ])
        self.assertTrue(all(spec.epochs == 1 for spec in loss_specs(protocol)))

    def test_selects_one_based_run_index(self):
        specs = loss_specs(self.protocol)
        self.assertEqual(_select_specs(specs, 2), [specs[1]])
        with self.assertRaisesRegex(ValueError, "entre 1 e 3"):
            _select_specs(specs, 0)
        with self.assertRaisesRegex(ValueError, "entre 1 e 3"):
            _select_specs(specs, 4)

    def test_merge_keeps_protocol_order_and_replaces_existing_result(self):
        specs = loss_specs(self.protocol)
        old = [{"name": specs[1].name, "map50": 0.1}]
        new = [
            {"name": specs[0].name, "map50": 0.2},
            {"name": specs[1].name, "map50": 0.3},
        ]
        merged = _merge_runs(old, new, specs)
        self.assertEqual([row["name"] for row in merged], [specs[0].name, specs[1].name])
        self.assertEqual(merged[1]["map50"], 0.3)

    def test_status_lists_pending_loss_runs_before_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            status = protocol_status("configs/colab_protocol.json", directory)
        self.assertEqual(status["phases"]["loss"]["pending_indices"], [1, 2, 3])
        self.assertEqual(status["phases"]["loss"]["next_index"], 1)
        self.assertTrue(status["phases"]["search"]["blocked"])

    @patch("plantdoc_tcc.staged.audit_dataset")
    @patch("plantdoc_tcc.staged._run_spec")
    def test_individual_loss_runs_persist_and_unlock_search(self, mocked_run, mocked_audit):
        mocked_audit.return_value = SimpleNamespace(invalid_lines=[])

        def fake_run(data, protocol, phase_dir, spec, device):
            score = {"baseline": 0.2, "class_weighted": 0.4, "focal": 0.3}[
                spec.loss_strategy
            ]
            return {
                **asdict(spec),
                "name": spec.name,
                "phase": phase_dir.name,
                "map50_95": score,
                "map50": score + 0.1,
                "weights": f"{spec.name}/best.pt",
                "results_csv": f"{spec.name}/results.csv",
            }

        mocked_run.side_effect = fake_run
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 2, 3):
                summary = run_phase(
                    "data.yaml",
                    "configs/colab_protocol.json",
                    directory,
                    "loss",
                    "cpu",
                    index,
                )
            status = protocol_status("configs/colab_protocol.json", directory)

        self.assertEqual(len(summary["runs"]["loss"]), 3)
        self.assertEqual(summary["winning_loss"], "class_weighted")
        self.assertFalse(status["phases"]["search"]["blocked"])
        self.assertEqual(status["phases"]["search"]["pending_indices"], list(range(1, 9)))


if __name__ == "__main__":
    unittest.main()
