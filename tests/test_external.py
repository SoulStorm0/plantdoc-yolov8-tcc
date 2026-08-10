import csv
import tempfile
import unittest
from pathlib import Path

from plantdoc_tcc.evaluate import validate_external_metadata


class ExternalProtocolTest(unittest.TestCase):
    def test_requires_adjudicated_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "images" / "leaf.jpg").write_bytes(b"image")
            (root / "labels" / "leaf.txt").write_text("0 .5 .5 .2 .2", encoding="utf-8")
            csv_path = root / "metadata.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["image", "lighting", "annotator_1", "annotator_2", "adjudicated"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image": "images/leaf.jpg",
                        "lighting": "direta",
                        "annotator_1": "A",
                        "annotator_2": "B",
                        "adjudicated": "false",
                    }
                )
            with self.assertRaisesRegex(ValueError, "não adjudicada"):
                validate_external_metadata(csv_path, root)


if __name__ == "__main__":
    unittest.main()

