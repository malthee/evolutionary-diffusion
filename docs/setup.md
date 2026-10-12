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
