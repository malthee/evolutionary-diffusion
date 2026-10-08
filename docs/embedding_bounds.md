# SDXL-Turbo embedding bounds: DiffusionDB versus Parti Prompts

This page retains the result tables and interpretation from the completed
`clamp-bound-comparison` study (7 October 2026), including its final local publication
report. The earlier committed report used a 54,326-prompt MPS sample; it is not the
full-corpus result. Experiment runners, notebooks, tensor snapshots, raw images and
operational receipts remain outside this documentation import.

## Findings

All **1,528,510 unique nonblank DiffusionDB 2M prompts** were processed. Relative to all
1,632 Parti prompts re-encoded on CUDA, token summed spans changed by **+112.686%** and
pooled summed spans by **+45.569%**. These are interval-width statistics, not semantic
coverage or hypervolume.

## Methods and provenance

The pinned metadata contained 2,000,000 rows; 603 blank rows and 470,887 exact-text
duplicates were excluded. Original retained text was not rewritten or language-filtered.
The unique text corpus was encoded once in the main pass, with extra pilot/parity/smoke
probes. Distinct texts can share the same truncated token sequence; deduplication
occurred before tokenization.

SDXL-Turbo revision `71153311d3dbb46851df1931d3ca6e939de83304`; FP16 CLIP-L/CLIP-G
encoding; 77-position padding/truncation; concatenated penultimate hidden states
(77×2048) and CLIP-G pooled projection (1280). Extrema were reduced in FP32. The tables
in this report recompute aggregates in FP64 from saved FP32 endpoints; small last-digit
differences from the runtime CSV are summation precision, not new encodings.

Encoding used single-prompt CUDA graph forwards, with outputs grouped in batches of 64
for transfer/reduction. This is not a batched model forward. The full corpus had 66,146
truncated prompts for CLIP-L and 67,783 for CLIP-G; padding/truncation matches the
77-position production representation.

Encoding disabled reduced-precision GEMM reductions, used the SDPA math backend with
FP32 intermediate attention calculations, and retained FP16 parameters and outputs.
Original CUDA kernel settings were restored before the calibrated image-generation
recipe. These are recorded backend controls; the tensor representation, tokenizer and
model parameters were unchanged.

Primary comparisons use fresh CUDA Parti. Earlier dataset studies use their own fresh
MPS Parti baseline and are labelled separately. The study retained the packaged Parti
tensors unchanged; its image experiment used them as the original initialization
condition and its tables use them as a numerical-drift reference.

## Full DiffusionDB versus CUDA Parti

| Dataset | Tensor | Coordinates | Min | Max | Σ spans | Δ spans (%) | Outside Parti (%) |
|---|---|---|---|---|---|---|---|
| Parti | token | all | -809 | 854.5 | 5.424e+05 | 0 | 0 |
| Parti | token | positions 1–76 | -35.5 | 25.109 | 5.424e+05 | 0 | 0 |
| Parti | token | CLIP-L | -809 | 854.5 | 2.1391e+05 | 0 | 0 |
| Parti | token | CLIP-G | -66.375 | 25.109 | 3.2849e+05 | 0 | 0 |
| Parti | pooled | all | -8.0625 | 7.8867 | 7,364 | 0 | 0 |
| DiffusionDB 2M unique | token | all | -809 | 854.5 | 1.1536e+06 | 112.69 | 98.699 |
| DiffusionDB 2M unique | token | positions 1–76 | -48.375 | 182.5 | 1.1536e+06 | 112.69 | 99.998 |
| DiffusionDB 2M unique | token | CLIP-L | -809 | 854.5 | 3.334e+05 | 55.859 | 98.7 |
| DiffusionDB 2M unique | token | CLIP-G | -66.375 | 182.5 | 8.2021e+05 | 149.69 | 98.699 |
| DiffusionDB 2M unique | pooled | all | -8.8516 | 9.0234 | 10,720 | 45.569 | 100 |

A coordinate is outside Parti when its observed lower endpoint is lower or its upper
endpoint is higher. This measures interval extension; it does not count prompts outside
the Parti distribution.

## All evaluated datasets

| Dataset | Backend | Encoded prompts | Tensor | Min | Max | Σ spans | Δ reference Parti (%) |
|---|---|---|---|---|---|---|---|
| Parti | CUDA | 1,632 | token | -809 | 854.5 | 5.424e+05 | 0 |
| Parti | CUDA | 1,632 | pooled | -8.0625 | 7.8867 | 7,364 | 0 |
| DiffusionDB 2M unique | CUDA | 1,528,510 | token | -809 | 854.5 | 1.1536e+06 | 112.69 |
| DiffusionDB 2M unique | CUDA | 1,528,510 | pooled | -8.8516 | 9.0234 | 10,720 | 45.569 |
| Parti packaged | original | 1,632 | token | -809 | 854.5 | 5.4218e+05 | -0.041313 |
| Parti packaged | original | 1,632 | pooled | -8.0625 | 7.8828 | 7,362.3 | -0.023353 |
| parti | MPS | 1,632 | token | -809 | 854.5 | 5.4246e+05 | 0 |
| parti | MPS | 1,632 | pooled | -8.0625 | 7.8828 | 7,363.4 | 0 |
| genai_bench | MPS | 1,600 | token | -809 | 854.5 | 5.0705e+05 | -6.5275 |
| genai_bench | MPS | 1,600 | pooled | -7.3359 | 6.9297 | 7,244.5 | -1.6146 |
| t2i_compbench | MPS | 5,952 | token | -809 | 854.5 | 4.9998e+05 | -7.8315 |
| t2i_compbench | MPS | 5,952 | pooled | -8.3594 | 7.543 | 7,679.9 | 4.2992 |
| diffusiondb sample | MPS | 54,326 | token | -809 | 854.5 | 9.2728e+05 | 70.939 |
| diffusiondb sample | MPS | 54,326 | pooled | -7.8945 | 8.3047 | 9,280.7 | 26.038 |
| drawbench | MPS | 200 | token | -809 | 854.5 | 4.1035e+05 | -24.353 |
| drawbench | MPS | 200 | pooled | -6.8477 | 6.457 | 5,867.4 | -20.316 |
| geneval | MPS | 553 | token | -809 | 854.5 | 2.8023e+05 | -48.341 |
| geneval | MPS | 553 | pooled | -7.1016 | 7.0742 | 6,025.7 | -18.167 |

The earlier DiffusionDB MPS entry is a 54,326-prompt sample. The CUDA entry covers every
retained prompt from the pinned 2M metadata, not every prompt in the larger DiffusionDB
14M release. MPS-versus-CUDA dataset differences are not controlled hardware comparisons
because corpus size and order differ.

## Numerical drift of the Parti baseline

| Comparison | Tensor | Δ summed spans (%) | Median absolute endpoint Δ | p95 absolute endpoint Δ | Max absolute endpoint Δ |
|---|---|---|---|---|---|
| CUDA versus packaged original | token | 0.04133 | 0.00097656 | 0.0039062 | 2.8379 |
| CUDA versus packaged original | pooled | 0.023358 | 0.00097656 | 0.0039062 | 0.42188 |
| CUDA versus fresh MPS | token | -0.010739 | 0.00097656 | 0.0039062 | 0.125 |
| CUDA versus fresh MPS | pooled | 0.0090914 | 0 | 0.0039062 | 0.0078125 |

These comparisons use the same prompt set but are not a controlled hardware-only
experiment: software, batch size and packaged-tensor provenance may also differ. Small
endpoint perturbations can change the strict outside-interval percentages. The matched
CUDA comparison is the primary result.

## Equal-count dataset comparison

| Tensor | Prompts | Replicates | Mean Δ (%) | SD (points) | Min Δ (%) | Max Δ (%) |
|---|---|---|---|---|---|---|
| token | 1,632 | 10 | 21.784 | 0.41979 | 20.92 | 22.399 |
| pooled | 1,632 | 10 | -0.026233 | 0.19058 | -0.3069 | 0.31521 |

Ten DiffusionDB samples of 1,632 prompts each are compared with the complete
1,632-prompt Parti dataset. Variability is across seeded random samples; Parti was not
resampled and these ranges are not confidence intervals for semantic diversity.

Earlier MPS studies used one equal-size pair of subsets per dataset:

| Dataset | Tensor | Prompts per dataset | Δ spans (%) | Outside MPS Parti (%) |
|---|---|---|---|---|
| genai_bench | token | 1,600 | -6.3594 | 65.905 |
| genai_bench | pooled | 1,600 | -1.4494 | 74.922 |
| t2i_compbench | token | 1,632 | -17.966 | 45.489 |
| t2i_compbench | pooled | 1,632 | -3.9141 | 65.312 |
| diffusiondb | token | 1,632 | 22.309 | 88.687 |
| diffusiondb | pooled | 1,632 | 0.26933 | 78.672 |
| drawbench | token | 200 | 4.343 | 78.84 |
| drawbench | pooled | 200 | -2.1036 | 72.812 |
| geneval | token | 553 | -39.231 | 26.852 |
| geneval | pooled | 553 | -10.455 | 54.922 |

These earlier single subsets have no replicate-based uncertainty estimate and use a
separate MPS reference. They complement the ten-replicate CUDA DiffusionDB study rather
than being interchangeable with it.

## Recovery of full-corpus intervals

| Tensor | Sample size | Mean recovery (%) | SD (points) | Min (%) | Max (%) | Mean endpoint p95 error |
|---|---|---|---|---|---|---|
| token | 1,632 | 57.26 | 0.19738 | 56.854 | 57.549 | 3.1444 |
| token | 5,000 | 64.902 | 0.070918 | 64.805 | 64.993 | 2.672 |
| token | 10,000 | 69.422 | 0.056049 | 69.337 | 69.478 | 2.3963 |
| token | 25,000 | 75.24 | 0.091567 | 75.124 | 75.454 | 2.0408 |
| token | 50,000 | 79.458 | 0.08428 | 79.346 | 79.639 | 1.783 |
| token | 100,000 | 83.657 | 0.12494 | 83.428 | 83.847 | 1.5238 |
| pooled | 1,632 | 68.678 | 0.13092 | 68.485 | 68.913 | 2.1413 |
| pooled | 5,000 | 74.789 | 0.11921 | 74.635 | 75.013 | 1.7722 |
| pooled | 10,000 | 78.374 | 0.14797 | 78.193 | 78.538 | 1.5725 |
| pooled | 25,000 | 82.859 | 0.17355 | 82.611 | 83.163 | 1.3312 |
| pooled | 50,000 | 86.129 | 0.10826 | 85.907 | 86.287 | 1.143 |
| pooled | 100,000 | 89.251 | 0.085289 | 89.078 | 89.346 | 0.96732 |

Recovery is Σ subset widths / Σ full-corpus widths. Endpoint errors compare subset and
full endpoints in embedding units. Seeds 420–429 each define nested samples, without
replacement, at all six sizes. Nested samples share observations; full-corpus extrema
are a finite-corpus reference rather than universal encoder bounds.

## Token positions, absolute positions and endpoint differences

Token positions are zero-based; coordinate indices refer to the concatenated 77×2048
tensor (768 CLIP-L plus 1280 CLIP-G features). Maximum absolute endpoint is `max(abs(min), abs(max))`; midpoint is `(min+max)/2`.
Pooled embeddings have no token position; −1 in the
table below denotes that absence. The ten largest span increases are shown for each
tensor.

Position zero is the beginning-of-text position and can dominate global min/max despite
contributing little or no interval width. Main tables retain it; the positions 1–76 comparison exposes the remaining ranges.
Padding and truncation
mean later token positions do not have identical linguistic roles across prompts.

Largest absolute span differences versus CUDA Parti:

| Tensor | Position | Coordinate | DDB min | DDB max | Parti min | Parti max | Δ span |
|---|---|---|---|---|---|---|---|
| token | 51 | 1,828 | -24.797 | 181.25 | -2.25 | 11.453 | 192.34 |
| token | 76 | 1,828 | -14 | 182.5 | -1.8672 | 5.2344 | 189.4 |
| token | 54 | 1,828 | -25.422 | 176.62 | -2.3203 | 11.797 | 187.93 |
| token | 64 | 1,828 | -23.938 | 172.25 | -2.3359 | 14.195 | 179.66 |
| token | 29 | 1,828 | -27.047 | 172.5 | -15.375 | 14.188 | 169.98 |
| token | 71 | 1,828 | -21.922 | 146.38 | -2.9922 | 13.438 | 151.87 |
| token | 41 | 1,828 | -26.406 | 143.62 | -5.3516 | 13.797 | 150.88 |
| token | 57 | 1,828 | -25.078 | 134 | -2.4922 | 14.344 | 142.24 |
| token | 63 | 1,828 | -24.391 | 123.62 | -2.4609 | 6.375 | 139.18 |
| token | 66 | 1,828 | -23.422 | 122.12 | -2.5938 | 6.25 | 136.7 |
| pooled | -1 | 834 | -6.9258 | 8.125 | -4.2969 | 4.4883 | 6.2656 |
| pooled | -1 | 359 | -5.8164 | 7.7031 | -2.5801 | 4.8555 | 6.084 |
| pooled | -1 | 340 | -4.8828 | 6.8594 | -2.3555 | 3.5703 | 5.8164 |
| pooled | -1 | 1,124 | -6.7344 | 6.7305 | -3.1191 | 4.5547 | 5.791 |
| pooled | -1 | 916 | -6.5078 | 7.5664 | -3.4512 | 4.8906 | 5.7324 |
| pooled | -1 | 236 | -7.9336 | 6.0586 | -4.1562 | 4.1172 | 5.7188 |
| pooled | -1 | 651 | -6.3828 | 8.25 | -4.2656 | 4.7305 | 5.6367 |
| pooled | -1 | 650 | -5.3359 | 7.7461 | -2.4453 | 5.0234 | 5.6133 |
| pooled | -1 | 814 | -6.1523 | 7.1133 | -4.0391 | 3.6738 | 5.5527 |
| pooled | -1 | 1,224 | -7.1719 | 8.2109 | -3.6328 | 6.2188 | 5.5312 |

## Paired synthetic initialization images

Exactly 100 DiffusionDB and 100 original packaged-Parti images were generated. Each pair
uses identical independently drawn FP32 uniform token/pooled coordinates, scaled into
each condition's bounds and cast to FP16. Diffusion seeds are 10,000–10,099; 512×512,
three inference steps, guidance zero, one image per embedding, matching batch positions.
The study retained all pairs and their sampled tensors, seeds and image checksums. These
artifacts are not part of main documentation.

| Condition | Pairs/images | Mean score | SD | Median | Min | Max |
|---|---|---|---|---|---|---|
| DiffusionDB | 100 | 5.983 | 0.51507 | 6.0553 | 4.8764 | 7.135 |
| Parti packaged | 100 | 5.9776 | 0.54742 | 5.9825 | 4.059 | 7.3239 |
| Paired difference | 100 | 0.0053357 | 0.67398 | -0.058462 | -1.6357 | 2.4368 |

DiffusionDB had the larger aesthetics score in 47/100 pairs, smaller in 53/100, tied in
0/100. Scoring device: cuda. The scorer is calibrated LAION improved aesthetic V2. These
descriptive scores test this specific initialization recipe; they do not establish human
preference, semantic faithfulness, or improved evolutionary outcomes.

## Interpretation and limits

Wider DiffusionDB intervals are plausible given its much larger corpus and broader user
prompt styles. Equal-count comparisons separate part of the dataset-composition effect
from sample-count effects. Exact extrema are tail-sensitive; ranges alone do not
describe density, coordinate correlation, occupied volume or sample quality.

Axis-aligned box hypervolume is the product of coordinate widths, whereas the reported
summed spans are their sum. Any box-volume or geometric-mean width ratio restricted to
common-variable coordinates uses a pair-specific mask; ratios with different masks are
not comparable. Constant coordinates make full ambient volume zero. Box measures depend
on the basis and include combinations not produced by real prompts. Independently
sampled synthetic embeddings deliberately discard correlations.

The mathematical encoder mapping is hardware-independent in exact arithmetic. Actual
FP16 outputs can depend on kernels, batch shape, backend and software versions. Fresh
CUDA Parti controls the main dataset comparison; packaged and fresh MPS Parti are drift
references, not bitwise reproduction guarantees. Per-coordinate drift is included in the
tables. The study did not automatically replace the original Parti clamp tensors;
importing this document changes no tensors or configuration.

## Using these results in the repository

The original `SDXLTurboEmbeddingRange` and `SDXLTurboPooledEmbeddingRange` still load
the packaged Parti tensors. The separate `load_embedding_bounds` loader supports
`source="diffusiondb"` and `source="parti"`; the reusable GA/OSGA examples select the
packaged full-corpus DiffusionDB bounds. See [campaign
configuration](configuration.md#recipe-and-optional-controls). This is an explicit
initialization choice, not a conclusion that wider bounds improve optimization. The
paired study used **three inference steps**; current examples use one, so its image
scores are not a forecast for those campaigns.

The DiffusionDB asset's
[manifest](../evolutionary_prompt_embedding/tensors/diffusiondb-full-bounds.json)
records 1,528,510 prompts, FP16 encoding, FP32 extrema and model revision. Its
Safetensors SHA-256 is
`301f5b3d453a035a6731ef7cf4c8c73511d22d95d02d294fdfc61e7e1e247cd2`. The documentation
import does not add a new bounds asset or alter this one.

## Recorded sources and provenance

These are the dataset revisions recorded by the study, not claims about current upstream
versions. No dataset text or new inference artifacts are included here.

| Dataset | Recorded source | Revision |
|---|---|---|
| Parti Prompts | [nateraw/parti-prompts](https://huggingface.co/datasets/nateraw/parti-prompts) | `944b156abfdad7627c3221b5ec4f6a6fb060a197` |
| DiffusionDB 2M | [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) | `fb620fbe49fa4420e0734bd9c0df11f51176b61f` |
| GenAI-Bench | [BaiqiL/GenAI-Bench-1600](https://huggingface.co/datasets/BaiqiL/GenAI-Bench-1600) | `e51e8a8070fadfbfe1f160436bf14e7664e87247` |
| T2I-CompBench | [NinaKarine/t2i-compbench](https://huggingface.co/datasets/NinaKarine/t2i-compbench) | `18f3f268acaf39c89f8a3afcc2b471243042c927` |
| DrawBench | [sayakpaul/drawbench](https://huggingface.co/datasets/sayakpaul/drawbench) | `ad794cba698c253ffdb6d18eff3fc87b004c1135` |
| GenEval | [djghosh13/geneval](https://github.com/djghosh13/geneval) | `af4902f24d3ca90ebbb446dd9891a59e0f82725f` |

Primary results were checked against the study's saved CUDA endpoints and paired score
rows when preparing this documentation. Reported aggregates use FP64 over FP32
endpoints; displayed values are rounded. Sampling SDs describe variation across the ten
replicates, not confidence intervals. Bounds, sample identities, full-precision tables
and the independent completion audit remain with the research artifacts. Reproducing the
aggregate table values does not reproduce model inference or establish universal encoder
limits.
