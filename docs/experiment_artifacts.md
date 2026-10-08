# Scientific precision and artifact contracts

## Precision and reproducibility

Set `NVIDIA_TF32_OVERRIDE=0` **before the kernel starts/imports PyTorch**. Also set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The notebook bootstrap checks startup and the
runtime disables both matmul and cuDNN TF32. CLIP ViT-L/14, feature normalization
and the LAION aesthetic V2 MLP run in FP32, evaluation mode and inference mode.
CUDA autocast is disabled while scoring. Default scalar aesthetic callers still
use CPU; the cache separates resolved device, checkpoint identity and precision.
Cached checkpoint paths must be immutable during a process lifetime.
Algorithm-equivalence tests use deterministic ordered evaluators; floating-point
CUDA results can depend on batch shape. Real reproducibility is checked within
the same configured batching recipe.

Preflight validates local checkpoint hashes, actual CUDA tensor computation, both
current/pending MIG modes, environment, offline pipeline reload and disk reserve.
Before evolution it generates 16 PNG references in chunks bounded by the
configured candidate batch size, compares CUDA against a separate
CPU process using the reference normalization, and requires MAE ≤.001 and maximum
error ≤.005. CPU and CUDA score vectors and the tested batch size are saved in `parity.json`.
A changed batch size requires a matching preflight without reloading compatible models. Model-file
hashes and installed versions are saved in every run. No installation happens
inside notebook execution. Keep host-specific environment evidence outside Git; installed versions and hashes
in each run describe its actual environment.

Batched GA is opt-in (`candidate_batch_size=1` keeps scalar behavior). Ordered
`create_solutions`/`evaluate_batch` methods have scalar fallbacks. Batched creators
and evaluators must not consume the evolutionary global RNG: use independent
candidate generators. Attempts are bounded by the batch limit, remaining
pressure/evaluation budget and the maximum of still-needed population attempts
and successes. This preserves the scalar stopping point, variation order,
strict threshold comparisons, success quota, elite reuse and lineage; it never
speculates extra evaluations. CPU tests compare trajectories and RNG states;
real same-seed repetitions also compare embeddings, PNGs, records and lineage.

## Scientific artifacts and optional Drive

Every fresh evaluated candidate, including rejected and interrupted-generation
attempts, has one canonical PNG under `attempts/images/`, original FP16 token and
pooled embeddings, fitness, parents, operator names, threshold and classification.
The `attempts/` Safetensors archive accepts multiple identified batches in one
target generation. Root archive records completed survivor populations separately;
elites reference their original evaluation's image. Join through a globally unique experiment UUID
and evaluation ID, never by population index alone. Existing archive readers
remain compatible. Generation boundaries drain writes and propagate failures.

Optional `buffered_attempt_writes=true` coalesces up to 32 immutable CPU
attempt snapshots per target generation and writes them with one background
archive owner. The existing archive schema and evaluation-ID joins are unchanged;
archive snapshot grouping and slot numbers may differ. Generation checkpoints
and final export wait for all attempts. Rejected and unfinished-generation
attempts are flushed on orderly termination or exceptions. Failed commits
propagate and retain uncommitted payloads in `pending_attempts.pt` where storage
permits. Abrupt process/host loss can lose at most 32 pending embedding attempts;
the append-only evaluation journal remains separate. Optimization resume is not
provided. This opt-in does not modify an already running kernel. CPU copying,
queue admission and drain waits are timed separately from background worker
time, which overlaps inference.

Outputs include evaluations, lineage, operator outcomes, generation statistics,
embedding summaries, fitness/pressure charts, stage timings, contact sheets,
separate survivor-best and evaluation-best GIFs (including unfinished attempts),
best-evaluation image references, environment/model/source manifests and portable CPU state
checkpoints. Checkpoints preserve population, RNG, frozen donor bank and history;
optimization resume is not implemented. PNG worker time overlaps inference and
is reported separately; core wall time includes callbacks/persistence, final
export time is separate. Use the [offline analysis notebook](../notebooks/embedding_analysis.ipynb) for
projections; additional evaluators are optional.

Finalization validates finite tensors/scores, evaluation identities, all PNGs,
counts and survivor joins. Only then, after saving the final executed notebook,
does the controller write `experiment_complete.json` and package an immutable
ZIP64 archive. A fully completed population-64/100-population run has 6,400 survivor records;
all fresh evaluations must be represented. Pressure/evaluation stopping can produce fewer completed populations. Interrupted runs retain evidence and are not
marked as completed artifacts.

`evolutionary_extensions.persistence.google_drive.GoogleDrivePersistor` exposes independently
callable `package`, `upload` and `cleanup`. Public Drive is disabled. A private
configuration supplies an existing dedicated `drive.file` OAuth credential
(mode 0600), expected account and accessible folder. Credentials and resumable
session URLs stay outside run archives. Test refresh plus a tiny upload/download
with deletion disabled first. The credential JSON fields are `client_id`,
`client_secret`, `refresh_token`, `scope` and `expected_account`. Upload resumes using a persisted file ID, avoiding
duplicate files and inference reruns. Failures preserve local outputs.

Cleanup requires live folder/account checks, remote size/MD5, streamed-download
SHA-256 and unchanged local file hashes inside the explicit runner root. Only the
matching run and ZIP are deleted; sanitized receipts, manifests and report copies
remain. Keep failed outputs to retry transfer. Never archive credentials, caches
or environment directories. Before rendering, reserve predicted space for the incoming batch, pending PNG
writes and next checkpoint. Persist an already evaluated batch if the deadline
or soft reserve crosses while inference is in flight; stop the next batch.
New inference stops below 20 GiB reserve; 8 GiB is
the hard floor. Consider projected archive space before a larger run.

## Exact parent reuse (opt-in)

`reuse_unchanged_offspring` defaults to false for historical recipe replay.
Enable it only with canonical scoring and a fixed physical render batch size.
Token and pooled tensors must both be exactly equal to a selected parent's
tensors; approximate similarity never reuses fitness. No creator or scorer call
is made for a match. Fresh evaluations, PNGs and embeddings keep their existing
archive format and contiguous IDs. Survivor records reference the original
evaluation/image. `reused_offspring.json` stores cached proposal classifications
and source IDs separately, including rejected and unfinished-generation proposals.
Checkpoints retain reused proposal records; result/progress report their count.
This is not a cache of arbitrary historical tensors or stochastic evaluations.
