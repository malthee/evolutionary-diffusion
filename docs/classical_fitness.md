# Classical image fitness

`evolutionary_imaging.classical_fitness.ClassicalImageEvaluator` implements CPU image measures for maximization-based search. It uses NumPy and Pillow; install the imaging extra in the usual package environment. It does not load a learned model or generate images.

```python
from PIL import Image
from evolutionary_imaging.classical_fitness import ClassicalImageEvaluator

evaluator = ClassicalImageEvaluator("intensity_entropy")
with Image.open("image.png") as image:
    result = evaluator.score_image(image)
print(result["fitness"], result["feasible"], result["components"])
```

The evaluator also accepts a creator result with exactly one PIL image in its `images` collection through `evaluate(result)` and returns a scalar fitness.

| Objective | Interpretation |
| --- | --- |
| `intensity_entropy` | Shannon entropy of arithmetic RGB intensity in bits. |
| `symmetry_liveliness` | Horizontal, vertical and diagonal symmetry combined with entropy. |
| `global_contrast_factor` | Multiscale contrast; defaults to the Matkovic 2005 recipe. |
| `fractal_target` | Edge box-counting dimension scored relative to target 1.35. |
| `benford_histogram` | Intensity histogram agreement using a disclosed L1 adaptation. |
| `ross_ralph_zong` | Gradient-response distribution distance, negated for maximization. |
| `machado_cardoso_jpeg2000` | Compression-based image and processing complexity. |
| `machado_cardoso_jpeg2000_regularized` | Separate compression variant requiring an explicit positive finite `stability_floor`. |

Every image is converted to a metadata-free 250 × 250 RGB raster with LANCZOS resizing. A PNG compressed-byte/raw-byte ratio below 0.08 is infeasible. Undefined scores are invalid. Invalid or infeasible images receive the finite rejection sentinel `REJECTED_FITNESS` (`-1e100`); inspect `valid`, `feasible`, `raw_score` and `signed_score` to distinguish them. Numerical overflow and scores outside the declared rejection domain raise errors.

Machado objectives require Pillow JPEG2000 support and fail at construction if it is missing. Regularization is an explicit variant, not a silent replacement for the original objective. `gcf_recipe="heijer2014_printed"` selects a diagnostic interpretation of the printed comparison formula; it is not interchangeable with the default.

Persist `evaluator.specification` with results to record measurement conventions, source locators and disclosed adaptations. These implementations do not establish exact equivalence with historical author software, and a higher objective score alone does not establish better perceptual image quality. Compare runs only with the same objective, specification and settings. The larger study controller and experimental plans are maintained separately from this public API.

CPU checks without model downloads:

```sh
python -m pytest tests/test_classical_fitness.py
```
