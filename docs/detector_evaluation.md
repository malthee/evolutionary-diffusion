# Five-detector image evaluation

This benchmark evaluates SAFE, UniversalFakeDetect (CLIP ViT-L/14 linear), SSP
(`sd4`), B-Free (`BFREE_dino2reg4`) and AEROBLADE. DIRE is excluded. It does not
train models or run evolutionary or multi-objective search.

## Frozen recipe

Seed 42 selects 200 calibration and 100 disjoint test photographs from RAISE-1k.
Preserve the original TIFFs; convert to RGB, resize the shorter side to 512 with
bicubic interpolation and center-crop to 512 square. Each canonical PNG has a
JPEG95 twin with subsampling disabled, derived from identical pixels.

Generate 100 uniformly sampled token/pooled embeddings using the full DiffusionDB
coordinate bounds and existing FP32 sampling / safe FP16 repair. Generate 100
ordinary controls using unchanged corresponding Synthbuster descriptions. All
200 generations are shared across detectors and retained without selection.
SDXL-Turbo revision `71153311d3dbb46851df1931d3ca6e939de83304` uses one step,
512 square, FP16, guidance zero and independently reset diffusion seed zero.
Generation uses four distinct candidates per batch. Conclusions are conditional
on this fixed-noise configuration.

The complete primary cross-product is **500 identities × two codecs × five
detectors = 5,000 scores**. Smoke, warm-up and batch profiling are separate.
SSP uses 64 seeded 32-square candidates, selects the simplest patch with the
official four-neighbour diversity rule and classifies once. Candidate coordinates
are shared between codecs; the selected patch may change after compression.
Preserve human probability and define synthetic score as one minus that value.
AEROBLADE preserves all three LPIPS-VGG layer-two reconstruction errors and uses
negative minimum error as its synthetic score. Other raw scales remain separate.

## Preparation and execution

Copy `configs/examples/detector_evaluation.json` into an ignored local directory
and set its absolute paths. Obtain official metadata with:

```sh
python -m evolutionary_extensions.experiments.detector_evaluation.metadata --help
python -m evolutionary_extensions.experiments.detector_evaluation.assets --output /absolute/assets
python -m evolutionary_extensions.experiments.detector_evaluation.runner prepare --config /absolute/config.json
```

The metadata helper reads only the prompt CSV and licence from the large
Synthbuster ZIP using validated HTTP ranges. The RAISE endpoint currently provides
its metadata ZIP without a registration submission. Preserve the downloaded
metadata, licences and hashes. Preparation freezes the split and CSV identities.

Model preparation pins official source commits and freezes checkpoint hashes,
model revisions and configuration. The original SD2 repository is unavailable;
its archived VAE weights and configuration are checked against two pinned copies.
The asset manifest records the download repository, both witnesses and the limit
that the original repository cannot currently be queried. This is an archived
copy of the chosen default model, not a different detector configuration.

Use separate benchmark environments and fresh managed kernels for each vendor;
their generic Python package names can otherwise collide. The released AEROBLADE
environment requires Python 3.11 and its older pinned stack; never install it into
the current experiment environment. Environment locks are under
`environments/detector-evaluation/`. Archive resolved package inventories too.

`pre-testing/detector_evaluation/detector_evaluation.ipynb` is a thin recipe interface. The managed
controller accepts a private `managed` configuration containing the existing
Jupyter URL, repository/root paths, per-stage kernels and lifecycle owner:

```sh
python -m evolutionary_extensions.experiments.detector_evaluation.managed --config /absolute/config.json
```

One worker runs generation, SAFE, UniversalFakeDetect, SSP, B-Free and AEROBLADE
sequentially. CUDA arithmetic, disabled current/pending MIG, A100 capacity,
available RAM and disk reserve are checked before inference. Set
`NVIDIA_TF32_OVERRIDE=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
Torch. Native inference parity, offline reload and repeatability are mandatory.
Profile batches 1/4/8/16 on the same 16-image pool for the first three classifiers;
AEROBLADE profiles 1/4. B-Free remains serial. Only batch sizes matching scalar
scores within absolute 1e-4 are accepted (AEROBLADE additionally permits relative
1e-3). Choose fastest validated throughput below 64 GiB peak reserved/allocated
memory. Cold load, preprocessing, inference and persistence are measured separately.

A separately authorized private owner handles Azure start/stop, deployment and
five-minute monitoring. Preserve the prior checkout/campaign, existing PT24H idle
fallback and disk reserve (20 GiB; hard floor 8 GiB). No time cutoff applies.
Notify once at ten hours elapsed or when projected runtime exceeds ten hours.
On terminal failure, retain partial evidence, close owned sessions and stop the
instance. A successful managed controller exit proves inference completion;
overall completion additionally requires report/archive checks and verified
Azure deallocation.

## Recovery and statistics

Scores are atomic records keyed by detector configuration, image hash and codec.
Changed inputs or configurations are rejected. Restarting processes unfinished
entries only. AEROBLADE additionally journals each component batch, reconstructions
and post-batch generator state, so recovery preserves its native sampling order.
Export or transfer failures must be retried without replaying completed inference.

Primary thresholds use only the 200 PNG calibration photos. Strict `score >
threshold` decisions handle ties conservatively and guarantee empirical FPR at
most 5%. Freeze thresholds before testing; use them unchanged for JPEG95 and
include secondary thresholds recalibrated on JPEG calibration photographs.

Compare real versus random and real versus prompted separately: AUROC, average
precision, detection/FPR, balanced accuracy and confusion counts. Rates use 95%
Wilson intervals. Two thousand seeded paired bootstrap resamples preserve
detector/codec pairing and real–prompted content pairs for ranking metrics and
detector differences. Matrix colours show PNG-calibration percentiles, never
probabilities. Undefined correlations are reported as unavailable.

## Reports and completion

Analysis/report stages consume exported evidence without GPU scoring:

```sh
python -m evolutionary_extensions.experiments.detector_evaluation.runner analyze --config /absolute/config.json
python -m evolutionary_extensions.experiments.detector_evaluation.runner report --config /absolute/config.json
python -m evolutionary_extensions.experiments.detector_evaluation.build_report --help
```

The interactive report uses the bundled Data app (React/Recharts), with a
300-image by five-detector matrix per codec, filters, full-size image links,
raw scores, decisions and SSP patch details. Scientific figures export PNG/PDF/SVG;
scores and metrics export CSV/JSON; findings export Markdown. Serve the local
report folder over HTTP so the split JSON assets load correctly.

Preserve images, originals, embeddings, prompts, splits, thresholds, coordinates,
source/model/environment provenance, executed stage notebooks and integrity
receipts. Recalculate values from the exported scores, exercise report filters,
inspect figures, verify archive checksums and record Azure `Stopped`. An unresolved
detector failure leaves the evaluation incomplete.

Sources: [RAISE guide](https://loki.disi.unitn.it/RAISE/guide.html),
[Synthbuster](https://zenodo.org/records/10066460),
[SSP patch procedure](https://github.com/bcmi/SSP-AI-Generated-Image-Detection/blob/main/utils/patch.py),
[AEROBLADE](https://github.com/jonasricker/aeroblade),
[SD2 archive](https://huggingface.co/sd2-community/stable-diffusion-2-base).
