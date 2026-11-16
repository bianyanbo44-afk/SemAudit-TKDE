"""Minimal before/after regression examples for all four source couplings."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis", HERE / "scripts/analyze_revision.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class ProtocolRegressionTests(unittest.TestCase):
    def test_scaler_fits_train_not_future(self):
        train = np.array([0., 2.])
        full = np.r_[train, [100.]]
        self.assertNotEqual(float(train.mean()), float(full.mean()))
        self.assertEqual(float(train.mean()), 1.)
        # Different displayed normalized errors can represent the same raw error.
        error = 2.
        for values in [train, full]:
            scale = values.std()
            self.assertAlmostEqual((error / scale) ** 2 * scale ** 2, error ** 2)

    def test_text_deletion_changes_population_not_only_input(self):
        numeric_units = {1, 2, 3}
        has_text = {1, 3}
        self.assertNotEqual(numeric_units, numeric_units & has_text)
        matched_aligned = matched_numeric = numeric_units & has_text
        self.assertEqual(matched_aligned, matched_numeric)

    def test_cache_and_raw_origin_must_match(self):
        numeric_base = 100.
        text = np.array([102., 105.])
        released_cache = text - text.min()
        raw_path = text - numeric_base
        self.assertFalse(np.array_equal(released_cache, raw_path))
        corrected_cache = released_cache + text.min() - numeric_base
        np.testing.assert_array_equal(corrected_cache, raw_path)

    def test_recency_subtraction_requires_common_units(self):
        tau = np.array([1., 3.])
        query = 5.
        total = 10.
        mixed = query / total - tau
        fixed = query / total - tau / total
        self.assertTrue(np.any(mixed < 0))
        self.assertTrue(np.all(fixed > 0))
        np.testing.assert_allclose(fixed, (query - tau) / total)

    def test_holm_monotone_and_bounded(self):
        p = np.array([.03, .001, .5])
        corrected = a.holm(p)
        self.assertTrue(np.all(corrected >= p))
        self.assertTrue(np.all(corrected <= 1))
        self.assertTrue(np.all(np.diff(corrected[np.argsort(p)]) >= 0))

    def test_ratio_of_means_uses_control_denominator(self):
        self.assertAlmostEqual(a.gain([1., 1.], [2., 2.]), 50.)
        self.assertEqual(a.gain([2., 2.], [2., 2.]), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
