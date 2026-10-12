import copy
import json
import os
import shutil
import subprocess

import pytest

from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
from evolutionary_extensions.experiments.detector_search.gallery import (
    HTML,
    matrix_rows,
)


def measurements():
    records = []
    for subject, noise, codec in (
        ("optimized-image", 0, "png"),
        ("control-image", 1, "jpeg95"),
    ):
        for detector in DETECTORS:
            records.append(
                {
                    "row": {
                        "subject": subject,
                        "group": "optimized" if noise == 0 else "random",
                        "target": "safe" if noise == 0 else None,
                        "diffusion_seed": noise,
                        "codec": codec,
                        "path": subject + ".png",
                        "sha256": subject,
                    },
                    "detector": detector,
                    "patch_seed": 42 if detector == "ssp" else None,
                    "threshold_png": 0.5,
                    "receipt_path": "measurement.json",
                    "value": {
                        "raw": 0.5,
                        "synthetic_score": 0.5,
                        "patch_path": "selected.png" if detector == "ssp" else None,
                        "selected_index": 0 if detector == "ssp" else None,
                        "coordinates": [[0, 0]] * 64 if detector == "ssp" else None,
                    },
                }
            )
    extra = copy.deepcopy(
        next(
            r
            for r in records
            if r["detector"] == "ssp" and r["row"]["diffusion_seed"] == 0
        )
    )
    extra["patch_seed"] = 43
    extra["value"]["raw"], extra["value"]["synthetic_score"] = 0.6, 0.4
    records.append(extra)
    return records


def test_percentile_is_not_probability_and_patch_changes_only_ssp():
    rows = matrix_rows(measurements(), {d: [0.5] * 200 for d in DETECTORS})
    selected = next(r for r in rows if r["diffusion_seed"] == 0)
    assert selected["scores"]["safe"]["42"]["calibration_percentile"] == 100
    assert selected["scores"]["safe"]["42"]["detected"] is False
    assert selected["scores"]["ssp"]["43"]["calibration_percentile"] == 0
    assert selected["scores"]["ssp"]["43"]["patch_coordinates"] == [[0, 0]] * 64
    with pytest.raises(ValueError, match="Duplicate"):
        matrix_rows(
            measurements() + measurements()[:1], {d: [0.5] * 200 for d in DETECTORS}
        )
    with pytest.raises(ValueError, match="Incomplete"):
        matrix_rows(measurements()[1:], {d: [0.5] * 200 for d in DETECTORS})
    with pytest.raises(ValueError, match="reference differs"):
        matrix_rows(measurements(), {d: [0.7] * 200 for d in DETECTORS})


def test_browser_filters_keep_missing_patch_measurements_explicit(tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the gallery's browser behavior test")
    rows = matrix_rows(measurements(), {d: [0.5] * 200 for d in DETECTORS})
    for row in rows:
        row.update(image_url="image.png", thumbnail_url="thumb.png")
        for pool in row["scores"].values():
            for score in pool.values():
                score.update(
                    receipt_url="receipt.json",
                    patch_url="patch.png" if score["patch_path"] else None,
                )
    payload = {"rows": rows, "detectors": list(DETECTORS)}
    script = HTML.split("</script><script>")[1].split("</script>")[0]
    fixture = tmp_path / "gallery-test.cjs"
    fixture.write_text(
        """const assert=require('node:assert/strict');
class Element { constructor(tag){this.tag=tag;this.children=[];this.style={};this.value='';this.textContent='';} append(...e){this.children.push(...e)} replaceChildren(){this.children=[]} addEventListener(){} }
const ids=Object.fromEntries(['payload','group','target','codec','noise','patch','head','rows','count'].map(k=>[k,new Element(k)]));
ids.payload.textContent=PAYLOAD;ids.group.value='all';ids.target.value='all';ids.codec.value='png';ids.noise.value='0';ids.patch.value='42';
const document={createElement:t=>new Element(t),getElementById:id=>ids[id]};
""".replace("PAYLOAD", json.dumps(json.dumps(payload)))
        + script
        + """
assert.equal(ids.rows.children.length,1);assert.equal(ids.rows.children[0].children.length,6);
ids.group.value='random';render();assert.equal(ids.rows.children.length,0);
ids.codec.value='jpeg95';ids.noise.value='1';ids.patch.value='43';render();assert.equal(ids.rows.children.length,1);
const cells=ids.rows.children[0].children;assert.equal(cells[3].textContent,'Unmeasured at this noise seed');assert.notEqual(cells[1].textContent,'Unmeasured at this noise seed');
ids.target.value='safe';render();assert.equal(ids.rows.children.length,0);
ids.target.value='all';ids.group.value='optimized';ids.codec.value='png';ids.noise.value='0';render();assert.equal(ids.rows.children.length,1);assert.equal(ids.rows.children[0].children[3].children[0].textContent,'0.0th calibration percentile');
"""
    )
    subprocess.run([node, str(fixture)], check=True, capture_output=True, text=True)
