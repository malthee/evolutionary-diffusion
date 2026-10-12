"""Local interactive held-out image matrix, colored by calibration percentiles."""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import quote

import numpy as np

from evolutionary_extensions.experiments.detector_evaluation.analysis import (
    threshold_at_fpr,
)
from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
from evolutionary_extensions.experiments.detector_evaluation.journal import (
    validate_complete,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .heldout_rendering import checked_path
from .reporting import load_measurements, read_json


def matrix_rows(records, calibration):
    if set(calibration) != set(DETECTORS) or any(
        len(v) != 200 or not all(math.isfinite(s) for s in v)
        for v in calibration.values()
    ):
        raise ValueError("Require all five frozen 200-image PNG calibration references")
    matrix = {}
    thresholds = {d: threshold_at_fpr(v) for d, v in calibration.items()}
    for record in records:
        row, value = record["row"], record["value"]
        if row["group"] == "calibration":
            continue
        if record["threshold_png"] != thresholds[record["detector"]]:
            raise ValueError(
                "Gallery calibration reference differs from the frozen PNG threshold"
            )
        key = (row["subject"], row["diffusion_seed"], row["codec"])
        item = matrix.setdefault(
            key,
            {
                "subject": row["subject"],
                "group": row["group"],
                "target": row["target"],
                "diffusion_seed": row["diffusion_seed"],
                "codec": row["codec"],
                "path": row["path"],
                "sha256": row["sha256"],
                "scores": {},
            },
        )
        if any(item[k] != row[k] for k in ("group", "target", "path", "sha256")):
            raise ValueError("Shared detector image identity differs")
        detector, patch = record["detector"], record["patch_seed"] or 42
        pool = item["scores"].setdefault(detector, {})
        if str(patch) in pool:
            raise ValueError("Duplicate gallery measurement")
        pool[str(patch)] = {
            "raw": value["raw"],
            "synthetic_score": value["synthetic_score"],
            "threshold_png": record["threshold_png"],
            "detected": value["synthetic_score"] > record["threshold_png"],
            "calibration_percentile": float(
                100
                * np.mean(np.asarray(calibration[detector]) <= value["synthetic_score"])
            ),
            "patch_path": value.get("patch_path"),
            "patch_coordinates": value.get("coordinates"),
            "selected_index": value.get("selected_index"),
            "receipt_path": record["receipt_path"],
        }
    for item in matrix.values():
        if set(item["scores"]) != set(DETECTORS) or any(
            "42" not in pool for pool in item["scores"].values()
        ):
            raise ValueError("Incomplete five-detector gallery row")
    return [matrix[key] for key in sorted(matrix)]


def calibration_references(search_root, benchmark_root):
    search_root, benchmark_root = Path(search_root), Path(benchmark_root)
    scores = read_json(benchmark_root / "reports/scores.json")
    images = read_json(benchmark_root / "images.json")
    if validate_complete(images, scores, DETECTORS) != 5000:
        raise ValueError("Benchmark calibration provenance is incomplete")
    references = {
        d: [
            r["synthetic_score"]
            for r in scores
            if r["detector"] == d
            and r["codec"] == "png"
            and r["group"] == "calibration"
        ]
        for d in DETECTORS
    }
    # Held-out AEROBLADE uses the separately calibrated per-image AE seed. The
    # benchmark's order-dependent AE samples are a different scoring procedure.
    path = search_root / "heldout/calibration/aeroblade-reused.json"
    adapted = read_json(path)
    references["aeroblade"] = [
        r["synthetic_score"] for r in adapted["receipts"] if r["codec"] == "png"
    ]
    return references, {
        "benchmark_scores_sha256": hashes(benchmark_root / "reports/scores.json")[
            "sha256"
        ],
        "aeroblade_calibration_receipt_sha256": hashes(path)["sha256"],
        "reference": "PNG real calibration empirical percentiles; adapted independent-seed AEROBLADE reference",
    }


HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Held-out detector matrix</title>
<style>body{font:15px system-ui;margin:24px;color:#202b37;background:#fafbfc}h1{font-size:25px}label{display:inline-block;margin:0 18px 12px 0}select{display:block;padding:7px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #cad4df;padding:10px;vertical-align:top}th{position:sticky;top:0;background:#eef2f6}td small{display:block}img{max-width:128px}details img{width:128px;image-rendering:pixelated}pre{max-width:320px;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}#scroll{overflow:auto}a{color:#185483}#count{font-weight:600}caption{text-align:left;padding:8px}</style>
<h1>Held-out detector image matrix</h1><p>Color: percentile relative to each detector's 200 PNG real calibration scores. <strong>Percentiles are not probabilities.</strong> Decisions use the unchanged PNG threshold; selected images do not estimate independent optimizer-run performance.</p>
<label>Group<select id="group"><option value="all">All</option><option>optimized</option><option>random</option><option>prompted</option></select></label>
<label>Optimized against<select id="target"><option value="all">All</option></select></label>
<label>Codec<select id="codec"><option>png</option><option>jpeg95</option><option>jpeg75</option></select></label>
<label>Diffusion noise seed<select id="noise"><option>0</option><option>1</option><option>2</option><option>3</option></select></label>
<label>SSP patch seed<select id="patch"><option>42</option><option>43</option><option>44</option><option>45</option></select></label>
<p id="count" aria-live="polite"></p><div id="scroll"><table><caption>Higher calibration percentile: more synthetic-oriented score. SSP seeds 43–45 were measured only at noise 0.</caption><thead id="head"></thead><tbody id="rows"></tbody></table></div>
<script id="payload" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),detectors=data.detectors;
const rawLabels={safe:'synthetic probability',universal_fake_detect:'synthetic probability',ssp:'human probability',bfree:'released fake logit',aeroblade:'minimum reconstruction error'};
const node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
const link=(label,url)=>{const e=node('a',label);e.href=url;e.target='_blank';e.rel='noopener';return e};
const select=document.getElementById('target');for(const d of detectors){const e=node('option',d);e.value=d;select.append(e)}
const header=node('tr');for(const title of ['Image',...detectors])header.append(node('th',title));document.getElementById('head').append(header);
function render(){const values=Object.fromEntries(['group','target','codec','noise','patch'].map(k=>[k,document.getElementById(k).value]));
const rows=data.rows.filter(r=>(values.group==='all'||r.group===values.group)&&(values.target==='all'||r.target===values.target)&&r.codec===values.codec&&String(r.diffusion_seed)===values.noise);
document.getElementById('count').textContent=`${rows.length} images × 5 detectors`;const body=document.getElementById('rows');body.replaceChildren();
for(const row of rows){const tr=node('tr'),image=node('td'),thumb=node('img');thumb.src=row.thumbnail_url;thumb.alt=row.subject;thumb.loading='lazy';image.append(thumb,node('small',`${row.subject} · ${row.group}`),link('Full image',row.image_url));tr.append(image);
for(const d of detectors){const td=node('td'),s=row.scores[d][d==='ssp'?values.patch:'42'];if(!s){td.textContent='Unmeasured at this noise seed';tr.append(td);continue}
td.style.backgroundColor=`hsl(210 55% ${96-s.calibration_percentile*0.35}%)`;td.append(node('strong',`${s.calibration_percentile.toFixed(1)}th calibration percentile`),node('small',s.detected?'Detected':'Not detected'),node('small',`Raw (${rawLabels[d]}): ${s.raw}`),node('small',`Synthetic score: ${s.synthetic_score}`),node('small',`PNG threshold: ${s.threshold_png}`));
const details=node('details');details.append(node('summary','Score / patch details'),link('Source receipt',s.receipt_url));if(s.patch_url){const patch=node('img');patch.src=s.patch_url;patch.alt=`${row.subject}: selected SSP patch`;details.append(node('p',`Selected candidate ${s.selected_index} of 64`),patch,node('pre',JSON.stringify(s.patch_coordinates)))}td.append(details);tr.append(td)}body.append(tr)}}
for(const k of ['group','target','codec','noise','patch'])document.getElementById(k).addEventListener('change',render);render();
</script></html>"""


def build_gallery(search_root, benchmark_root):
    from PIL import Image

    search_root = Path(search_root).resolve()
    records, evidence = load_measurements(search_root)
    calibration, provenance = calibration_references(search_root, benchmark_root)
    rows = matrix_rows(records, calibration)
    root = search_root / "reports"
    thumbnails = root / "thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = checked_path(search_root / "heldout", row["path"], row["sha256"])
        thumbnail = thumbnails / f"{row['sha256']}.png"
        if not thumbnail.exists():
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((128, 128))
                image.save(thumbnail)
        row.update(
            image_url="../heldout/" + quote(row["path"]),
            thumbnail_url="thumbnails/" + thumbnail.name,
        )
        for pool in row["scores"].values():
            for score in pool.values():
                score["receipt_url"] = "../" + quote(score["receipt_path"])
                score["patch_url"] = (
                    "../heldout/" + quote(score["patch_path"])
                    if score["patch_path"]
                    else None
                )
    payload = {
        "detectors": list(DETECTORS),
        "rows": rows,
        "calibration_provenance": provenance,
        "evidence": evidence,
    }
    atomic_json(root / "gallery_data.json", payload)
    serialized = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    path = root / "gallery.html"
    path.write_text(HTML.replace("__DATA__", serialized))
    atomic_json(
        root / "gallery_manifest.json",
        {
            "status": "gallery_exported",
            "html_sha256": hashes(path)["sha256"],
            "data_sha256": hashes(root / "gallery_data.json")["sha256"],
            "thumbnail_hashes": {
                p.name: hashes(p)["sha256"] for p in thumbnails.glob("*.png")
            },
            "rows": len(rows),
            "visual_audit_pending": True,
        },
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--benchmark-root", required=True)
    args = parser.parse_args()
    build_gallery(args.root, args.benchmark_root)


if __name__ == "__main__":
    main()
