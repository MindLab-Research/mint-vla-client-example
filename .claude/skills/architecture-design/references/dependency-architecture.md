# Dependency architecture

This repo now treats dependency management as two coordinated layers with one canonical metadata file:

1. The Mint worker image remains authoritative for ABI-bound packages and GPU runtime.
2. A PFS runtime-env root is authoritative for shared Python packages and pinned source overlays.

The metadata source of truth is:
- [pyproject.toml](/home/yiwen/tinker_project/tinker-server-dep-cleanup/pyproject.toml)
- `project.dependencies`: shared runtime Python packages
- `dependency-groups.host-runtime`: thin host-only additions
- `tool.tinker.runtime_env`: pinned source checkouts and runtime-env layout

## Why this split remains necessary

The worker image should continue owning packages whose correctness depends on CUDA, compiled extensions, or image build reproducibility:
- `torch`
- `vllm`
- `deep_ep`
- `deep_ep_cpp`
- `transformer_engine`

The PFS runtime-env root should own packages and sources that need to stay synchronized between the API host and Ray actors:
- API/driver Python packages such as FastAPI, Ray, OTEL, `requests`, `numpy`, `safetensors`
- pinned source trees for `verl`, `Megatron-Bridge`, and `Megatron-LM`

## Runtime-env root layout

Canonical layout under `PFS_RUNTIME_ENV_ROOT`:

```text
<root>/
  host-venv/
    bin/python
    bin/ray
  site-packages/
  src/
    Megatron-LM/
    Megatron-Bridge/
    verl/
  manifest.json
  activate_runtime_env.sh
```

Runtime import precedence:
1. `PFS_RUNTIME_ENV_ROOT/site-packages`
2. pinned source trees under `PFS_RUNTIME_ENV_ROOT/src`
3. `MINT_CODE_ROOT`
4. `PFS_HF_MODULES_PATH`

This precedence is surfaced as `tinker_server.config.PFS_PYTHONPATH`.

## Host startup model

The API host should launch with the runtime-env host interpreter:
- `<PFS_RUNTIME_ENV_ROOT>/host-venv/bin/python scripts/run_server.py`

`scripts/run_server.py` bootstraps `sys.path` and `PYTHONPATH` from `PFS_RUNTIME_ENV_ROOT` before importing the rest of the server. This removes the need to hand-assemble a long host `PYTHONPATH`.

## Worker startup model

Ray actors keep using `runtime_env={"env_vars": {...}}`, but the canonical env
now includes:
- `PFS_RUNTIME_ENV_ROOT`
- `MINT_CODE_ROOT`
- `PFS_HF_MODULES_PATH`
- `PYTHONPATH=PFS_PYTHONPATH`

No actor call site should hand-assemble per-package overlays or omit the
runtime-root variables.

There is no legacy overlay path in the supported design. The runtime must set
`PFS_RUNTIME_ENV_ROOT` and use the pinned sources materialized under that root.
