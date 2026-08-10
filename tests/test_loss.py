import importlib.util
import os
import unittest
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd()))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch não instalado no ambiente leve")
class LossTest(unittest.TestCase):
    def test_focal_reduces_easy_example_contribution(self):
        import torch

        from plantdoc_tcc.losses import focal_bce_elementwise

        targets = torch.tensor([[1.0, 0.0]])
        easy = torch.tensor([[8.0, -8.0]])
        hard = torch.tensor([[0.0, 0.0]])
        self.assertLess(focal_bce_elementwise(easy, targets).mean(), focal_bce_elementwise(hard, targets).mean())


if __name__ == "__main__":
    unittest.main()
