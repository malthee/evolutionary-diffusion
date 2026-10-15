"""Formula and invariance checks on deterministic rasters; no model inference."""

import math
import unittest

import numpy as np
from PIL import Image, features

from evolutionary_imaging.classical_fitness import (
    OBJECTIVES,
    REJECTED_FITNESS,
    ClassicalImageEvaluator,
    benford,
    box_dimension,
    entropy,
    feasible_ratio,
    gcf,
    measurement_image,
    regularize_machado,
    ross,
    symmetry,
)


class ClassicalFormulaTests(unittest.TestCase):
    def test_entropy_known_distributions_and_spatial_permutation(self):
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        self.assertEqual(entropy(ramp), 8.0)
        self.assertEqual(entropy(np.zeros((2, 2, 3), dtype=np.uint8)), 0.0)
        self.assertEqual(entropy(ramp[:, ::-1]), entropy(ramp))

    def test_symmetry_product_axes_and_strict_threshold(self):
        checker = np.repeat(np.array([[0, 255], [255, 0]])[:, :, None], 3, axis=2)
        score, components = symmetry(checker)
        self.assertEqual(components["horizontal"], 0.0)
        self.assertEqual(components["vertical"], 0.0)
        self.assertEqual(components["diagonal"], 1.0)
        self.assertAlmostEqual(score, 1 / 3)
        for delta, expected in [(12, 1.0), (13, 0.0)]:
            pixels = np.repeat(np.array([[0, delta], [0, delta]])[:, :, None], 3, axis=2)
            _, detail = symmetry(pixels)
            self.assertEqual(detail["horizontal"], expected)
        self.assertEqual(symmetry(np.full((4, 4, 3), 127))[0], 0.0)

    def test_benford_worst_probability_vertex_and_permutation(self):
        # The maximum L1 distance from Benford occurs at its least likely bin.
        value, detail = benford(np.full((4, 4, 3), 255, dtype=np.uint8))
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(detail["dmax"], 2 * (1 - math.log10(10 / 9)))
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        self.assertEqual(benford(ramp)[0], benford(ramp[:, ::-1])[0])
        self.assertAlmostEqual(sum(detail["expected"]), 1.0)

    def test_box_counting_known_point_line_and_plane_dimensions(self):
        point = np.zeros((256, 256), dtype=bool)
        point[0, 0] = True
        line = np.zeros_like(point)
        line[0] = True
        plane = np.ones_like(point)
        for mask, expected in [(point, 0), (line, 1), (plane, 2)]:
            self.assertAlmostEqual(box_dimension(mask)[0], expected, places=12)
        self.assertIsNone(box_dimension(np.zeros_like(point))[0])

    def test_original_gcf_analytic_black_white_checker(self):
        checker = np.repeat(np.array([[0, 255], [255, 0]])[:, :, None], 3, axis=2)
        value, detail = gcf(checker)
        # Only the size-one contrast is nonzero: lightness difference = 100.
        first_weight = (-0.406385 / 9 + 0.334573) / 9 + 0.0877526
        self.assertAlmostEqual(value, 100 * first_weight, places=12)
        self.assertEqual(detail["negative_weight_count"], 0)
        self.assertEqual(gcf(np.full((4, 4, 3), 127))[0], 0.0)

    def test_printed_gcf_is_labeled_and_keeps_the_published_conflict(self):
        checker = np.repeat(np.array([[0, 255], [255, 0]])[:, :, None], 3, axis=2)
        value, detail = gcf(checker, "heijer2014_printed")
        self.assertAlmostEqual(value, 10 * 0.334573 * 0.0877526, places=12)
        self.assertGreater(detail["negative_weight_count"], 0)
        self.assertEqual(detail["recipe"], "heijer2014_printed")
        self.assertNotEqual(value, gcf(checker)[0])

    def test_ross_unsigned_input_does_not_wrap_and_probabilities_sum_to_one(self):
        rng = np.random.default_rng(1729)
        raster = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
        score, components = ross(raster)
        self.assertEqual(score, ross(raster.astype(np.float64))[0])
        self.assertGreaterEqual(score, 0.0)
        self.assertAlmostEqual(components["histogram_probability_sum"], 1.0)
        self.assertAlmostEqual(components["normal_probability_sum"], 1.0)
        self.assertIsNone(ross(np.zeros((4, 4, 3), dtype=np.uint8))[0])

    def test_regularization_caps_equality_and_retains_original_invalidity(self):
        components = {"IC": 0.4, "PC0": 0.125, "PC1": 0.125,
                      "relative_stability": 0.0,
                      "invalid_reason": "zero processing/stability denominator"}
        value, detail = regularize_machado(components, 0.001)
        self.assertAlmostEqual(value, 25600)
        self.assertTrue(detail["stability_floor_active"])
        self.assertFalse(detail["legacy_raw_valid"])
        self.assertIsNone(detail["legacy_raw_score"])
        self.assertEqual(detail["legacy_invalid_reason"], components["invalid_reason"])
        self.assertNotIn("invalid_reason", detail)
        self.assertIn("invalid_reason", components)

    def test_regularization_is_identical_above_floor_and_rejects_invalid_pc(self):
        components = {"IC": 0.4, "PC0": 0.125, "PC1": 0.125,
                      "relative_stability": 0.01}
        value, detail = regularize_machado(components, 0.001)
        self.assertEqual(value, detail["legacy_raw_score"])
        self.assertFalse(detail["stability_floor_active"])
        self.assertIsNone(regularize_machado({**components, "PC0": 0}, 0.001)[0])
        for floor in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                regularize_machado(components, floor)

    def test_png_filter_boundary_and_metadata_free_measurement(self):
        self.assertTrue(feasible_ratio(0.08))
        self.assertFalse(feasible_ratio(math.nextafter(0.08, 0)))
        image = Image.new("RGB", (512, 512), "black")
        image.info["comment"] = "irrelevant metadata"
        measured = measurement_image(image)
        self.assertEqual(measured.size, (250, 250))
        self.assertEqual(measured.info, {})
        result = ClassicalImageEvaluator("intensity_entropy").score_image(image)
        self.assertEqual(result["raw_score"], 0)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["fitness"], REJECTED_FITNESS)

    def test_unrepresentable_floor_returns_invalid_instead_of_dividing_by_zero(self):
        components = {"IC": 0.4, "PC0": 0.125, "PC1": 0.125,
                      "relative_stability": 0.0}
        value, detail = regularize_machado(components, math.nextafter(0, 1))
        self.assertIsNone(value)
        self.assertIn("denominator", detail["invalid_reason"])

    @unittest.skipUnless(features.check("jpg_2000"), "JPEG2000 codec unavailable")
    def test_regularized_recipe_requires_an_explicit_floor_and_is_not_shared(self):
        with self.assertRaises(ValueError):
            ClassicalImageEvaluator("machado_cardoso_jpeg2000_regularized")
        evaluator = ClassicalImageEvaluator("machado_cardoso_jpeg2000_regularized",
                                            stability_floor=0.001)
        specification = evaluator.specification
        specification["machado_regularized"]["stability_floor"] = 0.5
        self.assertEqual(evaluator.specification["machado_regularized"]["stability_floor"], 0.001)

    @unittest.skipUnless(features.check("jpg_2000"), "JPEG2000 codec unavailable")
    def test_all_eight_score_saved_rasters_repeatably_and_orient_ross(self):
        rng = np.random.default_rng(144)
        image = Image.fromarray(rng.integers(0, 256, size=(250, 250, 3), dtype=np.uint8))
        for objective in OBJECTIVES:
            options = {"stability_floor": 0.001} if objective.endswith("regularized") else {}
            evaluator = ClassicalImageEvaluator(objective, **options)
            first = evaluator.score_image(image)
            self.assertEqual(first, evaluator.score_image(image))
            self.assertTrue(first["valid"])
            self.assertTrue(first["feasible"])
            self.assertTrue(math.isfinite(first["raw_score"]))
            self.assertEqual(first["signed_score"],
                             -first["raw_score"] if objective == "ross_ralph_zong" else first["raw_score"])


if __name__ == "__main__":
    unittest.main()
