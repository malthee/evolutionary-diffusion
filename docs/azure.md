# Prepare an existing Azure experiment host

The repository supplies portable experiment execution and an optional kernel
preparation command. Subscription, resource group, compute identity, idle settings,
credentials, campaign budgets and measured host capacity belong to private
configuration or external operations. See [architecture](architecture.md) and
[campaign configuration](configuration.md) for ownership and dependencies.

Copy a source checkout and its generic scientific inputs to the existing host.
Install inference/Jupyter dependencies in the selected environment before running
notebooks. The preparation command uses the active Python interpreter and writes
a disabled private profile plus a Jupyter kernel specification:

```sh
python -m evolutionary_extensions.azure.deployment \
  --repo "$PWD" --config configs/examples/osga.json \
  --cache "$HOME/.cache/evolutionary" --output "$PWD/.local/runs" \
  --jupyter-root "$PWD" --prepared-root "$PWD/.local/prepared/host"
```

Supply the actual Jupyter server root and output/cache locations for that host.
`--kernel-name` and `--kernel-root` can select the installation location; the
default uses the user's Jupyter kernels directory. Preparation preserves selected
bounds and scientific settings, refuses an existing prepared directory, keeps
execution disabled and does not enable deletion. An exported checkout may provide
`deployment_manifest.json` with `files` entries containing repository-relative
`path` and `sha256`; preparation verifies them before writing any settings.

The kernel sets `NVIDIA_TF32_OVERRIDE=0` before Torch, deterministic cuBLAS settings,
`EVOLUTIONARY_CHECKOUT`, `PYTHONPATH` and a noninteractive plotting backend. Startup
and CUDA/CPU parity checks still run in the notebook preflight; metadata preparation
is not GPU validation. Use the existing Azure-managed Jupyter server and its
configured authentication. No dependencies are installed, cells executed, resources
started/stopped or idle settings changed by preparation.

For optional Drive archival, supply `--credentials` and `--folder-id` together.
Keep credentials outside run/archive folders. Verify a nondeleting round trip
before enabling campaign execution; deletion remains an explicit private setting.
See [artifact and transfer contracts](experiment_artifacts.md).

An external lifecycle owner must stop compute on completion, failure or its
configured deadline, verify resource state and preserve unverified results.
The repository contains no machine-specific cleanup or shutdown supervisor.
