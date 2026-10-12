# Package setup

`pyproject.toml` defines package metadata, dependency ranges, extras and package
data. `setup.py` remains a small compatibility entry point for older tooling.
Pip and uv build the same setuptools package.

## uv

Use Python 3.12 or 3.13 for the current model stack. The core retains its Python
3.10 minimum. The checked-in `uv.lock` resolves the public extras and development
tools without installing every optional model dependency:

```sh
uv sync --locked --python 3.13 --extra cma
uv run --locked --extra cma pytest tests/test_sep_cma.py -q
```

For image-generation examples, select the existing domain extras:

```sh
uv sync --locked --extra imaging --extra prompt_embedding --extra execution
uv run --locked --extra imaging --extra prompt_embedding --extra execution python -m ipykernel install --user --name evolutionary
```

Core NumPy is declared directly. Domain extras reuse `model_helpers`; `all`
composes the public extras rather than maintaining a second dependency list.
The sound extra includes its imported diffusion stack. Hugging Face Hub and
PyYAML are explicit requirements for their consuming modules.

The `execution` extra supplies a Jupyter kernel and notebook execution tools, not
the notebook server. Launch notebooks with your existing Jupyter installation,
or install JupyterLab separately and select this kernel.

After an intentional dependency edit, run `uv lock` and review the lock diff.
Do not use `--upgrade` for a code cleanup. `uv lock --check` verifies consistency.
The inference, older vendor detector stacks and offline analysis environments
remain separate; their requirements are under `environments/`. The root lock
does not replace vendor pins or guarantee CUDA/model parity.

## Security baseline in 0.16.0

The optional model stack now requires PyTorch 2.14.1 / torchvision 0.29.1,
Diffusers 0.38, Transformers 5.17 and Datasets 5.0.1.
These release lines address eight root lockfile alerts and partially remediate
the Datasets alert;
fsspec 2026.6 and setuptools 83 are explicit security minimums, including
setuptools in isolated builds. The core API and Python 3.10 minimum are unchanged.
The Apple Silicon model extras require macOS 14 or newer; the patched Torch and
torchvision releases no longer provide macOS 11–13 wheels.

Datasets 5 changes streaming shuffle to mix multiple input shards. Set
`max_buffer_input_shards=1` on streaming shuffle to retain the earlier behavior.
PyTorch 2.13 removed named-tensor APIs such as `rename`, `refine_names` and
`align_to`; callers using those APIs must migrate. Transformers 5.17 is the first
checked release that gates custom-generation code downloads on trust consent.
Earlier advisory version ranges do not capture that boundary reliably. PyTorch 2.14 contains the TorchScript fix; the
reported 2.13 minimum still crashes on the original trigger.

Datasets 5.0.1 rejects the reported parent-directory, absolute and URL paths,
but a backslash metadata path variant still escapes the dataset directory on
POSIX. The latest 5.1 release retains that gap. Use trusted folder metadata;
the version update alone does not fully enforce containment.

Accelerate's sharded-checkpoint path traversal/denial-of-service alert remains
open. Its latest stable 1.15 release has no fix, and the proposed upstream patches
were closed without merging. The existing Accelerate requirement is retained;
use trusted checkpoint indexes while awaiting a complete upstream fix.

Recreate the selected environment from the lockfile and use fresh experiment
output directories with new dependency provenance. Repeat generator and detector
reference validation before comparing scientific results across these baselines.
The historical vendor and analysis recipes under `environments/` remain frozen
and require a separate compatibility review before use with untrusted inputs.

## pip

The existing editable installation remains supported:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[prompt_embedding,imaging,cma,execution]"
```

| Extra | Purpose |
|---|---|
| `cma` | Frozen pycma 4.4.4 used by the reusable Sep-CMA adapter and detector search |
| `imaging` | Image generation, evaluation and visualization |
| `model_helpers` | Torch/diffusers model helpers |
| `prompt_embedding` | Embedding creators, variation and Safetensors archives |
| `prompt_embedding_utils` | Dataset-backed embedding-bound utilities |
| `sound` | Audio creators/evaluators plus the shared model-helper stack |
| `execution` | Managed notebook execution and kernels |
| `all` | All public extras; excludes development tools and legacy vendor stacks |

## Validation and distributions

```sh
uv run --locked --extra cma pytest tests/test_sep_cma.py -q
uv run --locked ruff check <changed-python-files>
uv build
```

Domain tests need their domain extras. Offline projection/viewer tests need the
separate analysis requirements. Check wheel and source distribution assets after
packaging changes: embedding tensors and viewer HTML, detector report JSX/CSS,
and source-distribution notebook/configuration/environment examples. Private
`.local/` campaigns and generated results are excluded.

References: [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/),
[setuptools pyproject configuration](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).
