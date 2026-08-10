import json
import tempfile
import unittest
from pathlib import Path

from plantdoc_tcc.config import iter_specs, load_config


class ConfigTest(unittest.TestCase):
    def test_grid_size_and_unique_names(self):
        config = json.loads(Path("configs/experiments.json").read_text(encoding="utf-8"))
        specs = list(iter_specs(config))
        self.assertEqual(len(specs), 324)
        self.assertEqual(len({x.name for x in specs}), len(specs))

    def test_rejects_destructive_learning_rate(self):
        config = json.loads(Path("configs/experiments.json").read_text(encoding="utf-8"))
        config["lr0"] = [0.01]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "entre 1e-5 e 1e-3"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

