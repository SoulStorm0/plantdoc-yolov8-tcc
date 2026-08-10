import unittest

from plantdoc_tcc.data import class_weights, normalize_names


class DataTest(unittest.TestCase):
    def test_normalize_names_mapping(self):
        self.assertEqual(normalize_names({1: "b", 0: "a"}), ["a", "b"])

    def test_weights_prioritize_rare_classes_and_normalize(self):
        weights = class_weights([100, 25, 4])
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)

    def test_zero_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sem instâncias"):
            class_weights([10, 0])


if __name__ == "__main__":
    unittest.main()

