import csv
import json
import tempfile
import unittest
from pathlib import Path

from plantdoc_tcc.staged import load_protocol, loss_specs, score_results, search_specs


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


if __name__ == "__main__":
    unittest.main()
