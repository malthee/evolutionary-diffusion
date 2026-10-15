"""Audited classical scorers with explicit source and adaptation boundaries.

Scoring uses immutable 250-pixel RGB rasters and CPU float64 arithmetic. The
specification travels with each run; missing historical choices are exposed.
This editable revision does not replace frozen campaign implementations.
"""

import copy
import io
import math

import numpy as np
from PIL import Image, features

from evolutionary.evolution_base import SingleObjectiveEvaluator

OBJECTIVES = (
    "machado_cardoso_jpeg2000",
    "ross_ralph_zong",
    "symmetry_liveliness",
    "global_contrast_factor",
    "fractal_target",
    "intensity_entropy",
    "benford_histogram",
    "machado_cardoso_jpeg2000_regularized",
)
REJECTED_FITNESS = -1e100
SPECIFICATION = {
    "version": "classical-source-audit-v2",
    "claim": "verified formula families with disclosed conventions; exact den Heijer software equivalence unverified",
    "sources": {
        "comparison": "https://doi.org/10.1016/j.swevo.2014.01.002",
        "machado": "https://cdv.dei.uc.pt/wp-content/uploads/publications-cdv/mc98.pdf",
        "machado_bounds": "https://doi.org/10.1023/A:1013662402341",
        "gcf": "https://www.cg.tuwien.ac.at/research/publications/2005/matkovic-2005-glo/matkovic-2005-glo-PDF.pdf",
    },
    "measurement": {
        "size": [250, 250],
        "mode": "RGB",
        "resize": "Pillow LANCZOS",
        "dtype": "uint8",
    },
    "filter": {
        "format": "PNG",
        "compress_level": 6,
        "optimize": False,
        "metadata": "none",
        "denominator": "width*height*3 uint8 bytes",
        "reject": "encoded_bytes/raw_bytes < 0.08",
        "threshold": 0.08,
    },
    "entropy": {
        "intensity": "floor((R+G+B)/3)",
        "bins": 256,
        "log_base": 2,
        "normalization": "none",
        "source_locator": "den Heijer 2014 section 2.4 equation 12; arithmetic intensity equation 28",
        "fidelity": "Shannon formula verified; floor quantization and base two explicitly resolved",
    },
    "symmetry": {
        "intensity": "(R+G+B)/(3*255)",
        "strict_tolerance": 0.05,
        "axes": "left/right, top/bottom, opposite quadrants mirrored in both axes",
        "aggregation": "mean of horizontal, vertical, diagonal; multiply entropy bits",
        "source_locator": "den Heijer 2014 equations 23-32",
        "fidelity": "axes, strict tolerance and compound product verified; entropy conventions declared above",
    },
    "benford": {
        "bins": "floor(arithmetic_intensity*9/256), dark to light",
        "distance": "absolute L1",
        "p": 1,
        "normalization": "maximum possible L1=2*(1-min(log10(1+1/d)))",
        "adaptation": "absolute values, bin ordering and normalization explicitly resolved",
        "source_locator": "den Heijer 2014 equations 1-2, p=1",
        "fidelity": "printed signed p=1 difference cancels for probability histograms; L1 is a disclosed mathematical resolution, not recovered author code",
    },
    "fractal": {
        "mask": "forward x/y gradient magnitude of arithmetic intensity >0.05",
        "scales": [1, 2, 4, 8, 16, 32, 64, 128],
        "grid": "origin top-left; partial boundary boxes retained",
        "fit": "OLS log(count) against log(1/box_size)",
        "target": 1.35,
        "adaptation": "mask, scale ladder and grid are technical choices",
        "source_locator": "den Heijer 2014 equation 3",
        "fidelity": "target fitness verified; mask and box-counting conventions not specified by reference study",
    },
    "gcf": {
        "brightness": "0.299R+0.587G+0.114B",
        "gamma": 2.2,
        "scales": [1, 2, 4, 8, 16, 25, 50, 100, 200],
        "superpixel": "mean linear luminance; partial boundary cells retained",
        "lightness": "100*sqrt(linear_luminance)",
        "contrast": "mean available-neighbor absolute lightness difference",
        "weights": "(-0.406385*(i/9)+0.334573)*(i/9)+0.0877526; i=1..9",
        "adaptation": "original Matkovic lightness; no second square root or 2014 contrast exponent",
        "recipe": "matkovic2005",
        "source_locator": "Matkovic 2005 sections 3-5; conflicting den Heijer 2014 equations 4-11",
        "fidelity": "original formula primary; optional printed-2014 diagnostic is not verified author software",
    },
    "ross": {
        "gradient": "Roberts diagonal RGB gradient; uint8 channels as float64",
        "d": "0.001*hypot(width,height)/2",
        "response": "stimulus/2",
        "moments": "response-weighted mean and variance",
        "histogram": "response-weighted bins of sigma/100 with normal tail bins",
        "score": "1000*KL(p||q), natural log, minimize",
        "adaptation": "weighted histogram; cropped last row/column; q floor 1e-300",
        "source_locator": "den Heijer 2014 equations 17-22",
        "fidelity": "response, weighted moments, bin width and KL direction verified; histogram weighting/origin/tails are declared conventions",
    },
    "machado": {
        "jpeg": {"quality": 75, "subsampling": 0, "optimize": False},
        "IC": "RGB RMSE / (raw_bytes/JPEG_bytes)",
        "jpeg2000": {
            "quality_mode": "rates",
            "quality_layers": [8],
            "irreversible": True,
        },
        "PC0": "whole-image JPEG2000 bytes/raw_bytes",
        "PC1": "sum JPEG2000 bytes of four quadrants enlarged to 250 square / their total raw bytes",
        "score": "IC/((PC0*PC1)*(abs(PC1-PC0)/PC1))",
        "a_b_c": [1, 1, 1],
        "adaptation": "codec/quality mapping, encoded-fraction PC proxy, quadrant detail, exponents and absolute stability are explicit variants",
        "source_locator": "den Heijer 2014 equations 13-16 and JPEG2000 substitution",
        "fidelity": "legacy proxy retained for paired pathology ablation; PC omits reconstruction error from original compression-complexity definition",
    },
    "machado_regularized": {
        "score": "IC/(PC0*PC1*max(abs(PC1-PC0)/PC1, tau))",
        "stability_floor": None,
        "fidelity": "denominator-only adaptation inspired by Machado 2002 component bounds; not the published a=1,b=0.4,c=0.2 recipe",
        "policy": "tau must be supplied explicitly, justified on development data and frozen before fresh seed blocks",
    },
    "degenerate": "undefined score => raw null, invalid/infeasible, finite rejection fitness; no denominator epsilon",
    "fitness": "raw for maximization, negative raw for Ross minimization; infeasible=-1e100",
}


def measurement_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL raster")
    resized = image.convert("RGB").resize((250, 250), Image.Resampling.LANCZOS)
    # Encoder metadata must not become a way to pass the compression filter.
    return Image.fromarray(np.asarray(resized, dtype=np.uint8).copy())


def entropy(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    intensity = np.floor(rgb.mean(axis=2)).astype(np.uint8)
    counts = np.bincount(intensity.ravel(), minlength=256)
    p = counts[counts > 0] / intensity.size
    return float(-(p * np.log2(p)).sum())


def symmetry(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    values = rgb.mean(axis=2) / 255
    h, w = values.shape
    h2, w2 = h // 2, w // 2

    def similar(a, b):
        return float((np.abs(a - b) < 0.05).mean())

    horizontal = similar(values[:, :w2], values[:, -w2:][:, ::-1])
    vertical = similar(values[:h2], values[-h2:][::-1])
    diagonal = (
        similar(values[:h2, :w2], values[-h2:, -w2:][::-1, ::-1])
        + similar(values[:h2, -w2:], values[-h2:, :w2][::-1, ::-1])
    ) / 2
    lively = entropy(rgb)
    return (horizontal + vertical + diagonal) / 3 * lively, {
        "horizontal": horizontal,
        "vertical": vertical,
        "diagonal": diagonal,
        "entropy_bits": lively,
    }


def benford(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    indices = np.floor(rgb.mean(axis=2) * 9 / 256).astype(int)
    p = np.bincount(indices.ravel(), minlength=9) / indices.size
    expected = np.log10(1 + 1 / np.arange(1, 10))
    distance = float(np.abs(p - expected).sum())
    maximum = float(2 * (1 - expected.min()))
    return 1 - distance / maximum, {
        "histogram": p.tolist(),
        "expected": expected.tolist(),
        "L1": distance,
        "dmax": maximum,
    }


def box_dimension(mask):
    counts = []
    for size in SPECIFICATION["fractal"]["scales"]:
        boxes = np.add.reduceat(
            np.add.reduceat(
                mask.astype(np.int32), np.arange(0, mask.shape[0], size), axis=0
            ),
            np.arange(0, mask.shape[1], size),
            axis=1,
        )
        counts.append(int((boxes > 0).sum()))
    if not any(counts):
        return None, {"box_counts": counts, "invalid_reason": "empty edge mask"}
    x = -np.log(np.asarray(SPECIFICATION["fractal"]["scales"], dtype=float))
    y = np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.square(y - (slope * x + intercept)).sum())
    return float(slope), {
        "box_counts": counts,
        "log_fit_residual_sum_squares": residual,
    }


def fractal(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    intensity = rgb.mean(axis=2) / 255
    dx = np.diff(intensity, axis=1, append=intensity[:, -1:])
    dy = np.diff(intensity, axis=0, append=intensity[-1:])
    mask = np.hypot(dx, dy) > 0.05
    dimension, components = box_dimension(mask)
    components.update(dimension=dimension, edge_pixels=int(mask.sum()))
    return (
        None if dimension is None else max(0.0, 1 - abs(dimension - 1.35))
    ), components


def block_means(values, size):
    sums = np.add.reduceat(
        np.add.reduceat(values, np.arange(0, values.shape[0], size), axis=0),
        np.arange(0, values.shape[1], size),
        axis=1,
    )
    rows = np.minimum(size, values.shape[0] - np.arange(0, values.shape[0], size))
    cols = np.minimum(size, values.shape[1] - np.arange(0, values.shape[1], size))
    return sums / (rows[:, None] * cols[None, :])


def local_contrast(lightness):
    total, neighbors = np.zeros_like(lightness), np.zeros_like(lightness)
    for axis in (0, 1):
        left, right = [slice(None)] * 2, [slice(None)] * 2
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        delta = np.abs(lightness[left] - lightness[right])
        total[left] += delta
        total[right] += delta
        neighbors[left] += 1
        neighbors[right] += 1
    return float(
        np.divide(
            total, neighbors, out=np.zeros_like(total), where=neighbors > 0
        ).mean()
    )


def gcf(rgb, recipe="matkovic2005"):
    """Original GCF, or an explicitly labeled reading of the 2014 equations.

    The 2014 scale list omits size one despite a nine-term sum. The diagnostic
    completes that list with one and uses k=1..9. Its printed weight expression
    yields negative weights at coarse scales; it is not silently substituted
    for the original nonnegative contrast aggregation.
    """
    if recipe not in ("matkovic2005", "heijer2014_printed"):
        raise ValueError("Unknown GCF recipe")
    rgb = np.asarray(rgb, dtype=np.float64)
    brightness = rgb @ np.array([0.299, 0.587, 0.114])
    linear = (brightness / 255) ** 2.2
    lightness = [100 * np.sqrt(block_means(linear, size))
                 for size in SPECIFICATION["gcf"]["scales"]]
    if recipe == "matkovic2005":
        contrasts = [local_contrast(values) for values in lightness]
        x = np.arange(1, 10) / 9
        weights = (-0.406385 * x + 0.334573) * x + 0.0877526
        powers = np.ones(9)
    else:
        x = np.arange(9) / 8
        weights = (-0.406385 * x + 0.334573) * (x + 0.0877526)
        powers = 1 - np.arange(9) / 70
        contrasts = [local_contrast(np.sqrt(values)) ** power
                     for values, power in zip(lightness, powers)]
    return float(np.dot(weights, contrasts)), {
        "recipe": recipe,
        "scale_contrasts": contrasts,
        "weights": weights.tolist(),
        "powers": powers.tolist(),
        "negative_weight_count": int((weights < 0).sum()),
    }


def ross(rgb):
    # Public helper calls may receive uint8; subtraction must never wrap.
    rgb = np.asarray(rgb, dtype=np.float64)
    d = 0.001 * math.hypot(*rgb.shape[:2]) / 2
    first = rgb[:-1, :-1] - rgb[1:, 1:]
    second = rgb[1:, :-1] - rgb[:-1, 1:]
    response = (
        np.sqrt(np.square(first).sum(axis=2) + np.square(second).sum(axis=2)) / d / 2
    )
    total = float(response.sum())
    if total == 0:
        return None, {"invalid_reason": "zero stimulus"}
    mean = float(np.square(response).sum() / total)
    variance = float((response * np.square(response - mean)).sum() / total)
    sigma = math.sqrt(variance)
    if sigma == 0:
        return None, {"invalid_reason": "zero weighted variance", "mean": mean}
    width = sigma / 100
    low = math.floor(float(response.min()) / width) * width
    count = math.ceil((float(response.max()) - low) / width) + 1
    if count > 100000:
        raise ValueError("Unexpected response histogram size")
    edges = low + np.arange(count + 1) * width
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    p = np.histogram(response, bins=edges, weights=response)[0] / total
    z = (edges - mean) / sigma
    # erfc computes positive-tail masses more accurately than subtracting CDFs near 1.
    cdf = np.array([0.5 * math.erfc(-value / math.sqrt(2)) for value in z])
    survival = np.array([0.5 * math.erfc(value / math.sqrt(2)) for value in z])
    q = np.where(z[:-1] >= 0, survival[:-1] - survival[1:], cdf[1:] - cdf[:-1])
    active = p > 0
    value = float(
        1000 * (p[active] * np.log(p[active] / np.maximum(q[active], 1e-300))).sum()
    )
    return value, {
        "response_mean": mean,
        "response_sigma": sigma,
        "response_variance": variance,
        "bin_width": width,
        "bins": len(p),
        "histogram_counts": np.histogram(response, bins=edges)[0].tolist(),
        "response_weighted_probabilities": p.tolist(),
        "normal_probabilities": q.tolist(),
        "finite_bin_lower_origin": low,
        "response_sum": total,
        "q_floor_bins": int((q[active] < 1e-300).sum()),
        "histogram_probability_sum": float(p.sum()),
        "normal_probability_sum": float(q.sum()),
    }


def codec(image, fmt, **options):
    stream = io.BytesIO()
    image.save(stream, format=fmt, **options)
    data = stream.getvalue()
    with Image.open(io.BytesIO(data)) as decoded:
        decoded_rgb = np.asarray(decoded.convert("RGB"), dtype=np.float64)
    return len(data), decoded_rgb


def machado(image, rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    jpeg_bytes, decoded = codec(
        image, "JPEG", quality=75, subsampling=0, optimize=False
    )
    rms = float(np.sqrt(np.square(rgb - decoded).mean()))
    ratio = rgb.size / jpeg_bytes
    ic = rms / ratio
    opts = {"quality_mode": "rates", "quality_layers": [8], "irreversible": True}
    encoded, _ = codec(image, "JPEG2000", **opts)
    pc0 = encoded / rgb.size
    h, w = image.size
    quadrants = [
        (0, 0, h // 2, w // 2),
        (h // 2, 0, h, w // 2),
        (0, w // 2, h // 2, w),
        (h // 2, w // 2, h, w),
    ]
    detail_bytes = sum(
        codec(
            image.crop(box).resize(image.size, Image.Resampling.LANCZOS),
            "JPEG2000",
            **opts,
        )[0]
        for box in quadrants
    )
    pc1 = detail_bytes / (4 * rgb.size)
    stability = abs(pc1 - pc0) / pc1
    denominator = pc0 * pc1 * stability
    components = {
        "jpeg_bytes": jpeg_bytes,
        "jpeg_RMSE": rms,
        "JPEG_raw_over_encoded": ratio,
        "IC": ic,
        "PC0": pc0,
        "PC1": pc1,
        "relative_stability": stability,
        "JPEG2000_whole_bytes": encoded,
        "JPEG2000_detail_bytes": detail_bytes,
    }
    if denominator == 0:
        return None, {
            **components,
            "invalid_reason": "zero processing/stability denominator",
        }
    return ic / denominator, components


def regularize_machado(components, stability_floor):
    """Rescore stored components without introducing another codec or render.

    Zero stability is a valid capped case for this variant; nonpositive PC or
    invalid components remain invalid. The legacy score's validity is retained.
    """
    if not math.isfinite(stability_floor) or stability_floor <= 0:
        raise ValueError("Machado stability floor must be finite and positive")
    ic, pc0, pc1, stability = (components[key]
                             for key in ("IC", "PC0", "PC1", "relative_stability"))
    if (not all(math.isfinite(value) for value in (ic, pc0, pc1, stability))
            or ic < 0 or pc0 <= 0 or pc1 <= 0 or stability < 0):
        return None, {**components, "invalid_reason": "invalid compression components"}
    denominator = pc0 * pc1 * max(stability, stability_floor)
    if denominator <= 0 or not math.isfinite(denominator):
        return None, {**components, "invalid_reason": "unrepresentable regularized denominator"}
    score = ic / denominator
    if not math.isfinite(score):
        return None, {**components, "invalid_reason": "nonfinite regularized score"}
    legacy_denominator = pc0 * pc1 * stability
    legacy_score = ic / legacy_denominator if legacy_denominator > 0 else None
    legacy_valid = legacy_score is not None and math.isfinite(legacy_score)
    result = {key: value for key, value in components.items()
              if key != "invalid_reason"}
    result.update(
        stability_floor=float(stability_floor),
        stability_floor_active=bool(stability < stability_floor),
        regularized_denominator=float(denominator),
        legacy_raw_valid=legacy_valid,
        legacy_raw_score=legacy_score if legacy_valid else None,
        legacy_invalid_reason=components.get("invalid_reason"),
    )
    return score, result


def feasible_ratio(ratio):
    return ratio >= 0.08


class ClassicalImageEvaluator(SingleObjectiveEvaluator):
    def __init__(self, objective, *, gcf_recipe="matkovic2005", stability_floor=None):
        if objective not in OBJECTIVES:
            raise ValueError("Unknown historical objective")
        if objective.startswith("machado_cardoso") and not features.check("jpg_2000"):
            raise RuntimeError("Pillow JPEG2000 support is required")
        if gcf_recipe not in ("matkovic2005", "heijer2014_printed"):
            raise ValueError("Unknown GCF recipe")
        if objective == "machado_cardoso_jpeg2000_regularized":
            if stability_floor is None or not math.isfinite(stability_floor) or stability_floor <= 0:
                raise ValueError("Regularized Machado requires an explicit positive finite floor")
        elif stability_floor is not None:
            raise ValueError("Stability floor applies only to regularized Machado")
        self.objective = objective
        self.gcf_recipe = gcf_recipe
        self.stability_floor = stability_floor

    @property
    def specification(self):
        specification = copy.deepcopy(SPECIFICATION)
        specification["gcf"]["recipe"] = self.gcf_recipe
        if self.gcf_recipe == "heijer2014_printed":
            specification["gcf"].update(
                contrast="sqrt(lightness) neighbor differences; local mean raised to 1-(k-1)/70",
                weights="(-0.406385*x+0.334573)*(x+0.0877526), x=(k-1)/8, k=1..9",
                adaptation="printed diagnostic: size one added to reconcile eight listed sizes with nine-term sum; negative coarse weights retained",
            )
        specification["machado_regularized"]["stability_floor"] = self.stability_floor
        return specification

    def score_image(self, image):
        image = measurement_image(image)
        rgb = np.asarray(image, dtype=np.float64)
        stream = io.BytesIO()
        image.save(stream, format="PNG", compress_level=6, optimize=False)
        ratio = len(stream.getvalue()) / rgb.size
        if self.objective.startswith("machado_cardoso"):
            raw, components = machado(image, rgb)
            if self.objective == "machado_cardoso_jpeg2000_regularized":
                raw, components = regularize_machado(components, self.stability_floor)
        elif self.objective == "intensity_entropy":
            raw = entropy(rgb)
            components = {"entropy_bits": raw}
        elif self.objective == "global_contrast_factor":
            raw, components = gcf(rgb, self.gcf_recipe)
        else:
            raw, components = {
                "ross_ralph_zong": ross,
                "symmetry_liveliness": symmetry,
                "fractal_target": fractal,
                "benford_histogram": benford,
            }[self.objective](rgb)
        valid = raw is not None and math.isfinite(raw)
        if raw is not None and not valid:
            raise ValueError("Nonfinite raw score")
        signed = (
            -raw if raw is not None and self.objective == "ross_ralph_zong" else raw
        )
        if valid and abs(signed) >= 1e90:
            raise ValueError("Score outside declared finite rejection domain")
        feasible = valid and feasible_ratio(ratio)
        return {
            "objective": self.objective,
            "specification_version": SPECIFICATION["version"],
            "gcf_recipe": self.gcf_recipe if self.objective == "global_contrast_factor" else None,
            "stability_floor": self.stability_floor,
            "raw_score": raw,
            "signed_score": signed,
            "fitness": float(signed) if feasible else REJECTED_FITNESS,
            "valid": valid,
            "feasible": feasible,
            "png_ratio": ratio,
            "measurement_png_bytes": len(stream.getvalue()),
            "components": components,
        }

    def evaluate(self, result):
        if len(result.images) != 1:
            raise ValueError("Exactly one image per candidate required")
        return self.score_image(result.images[0])["fitness"]
