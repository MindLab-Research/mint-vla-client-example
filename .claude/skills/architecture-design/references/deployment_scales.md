# Deployment scales (image plus runtime-env root)

This repo uses two coordinated dependency surfaces. The canonical model is:

## 1) GPU worker image

Purpose:
- Runs on Volcano or Aliyun GPU tasks.
- Owns ABI-bound packages and compiled extensions.

Authoritative contents:
- CUDA stack
- `torch`
- `vllm`
- `deep_ep`
- `deep_ep_cpp`
- `transformer_engine`

Rule:
- Do not override these from arbitrary PFS checkouts.
- If a deployment needs to change one of these, rebuild the image instead of layering a partial package tree on top.

## 2) PFS runtime-env root

Purpose:
- Supplies one shared Python dependency graph and one set of pinned source trees for both the API host and Ray actors.

Authoritative contents:
- shared Python packages under `site-packages/`
- pinned source trees under `src/`
- thin host interpreter under `host-venv/`

Canonical root variable:
- `PFS_RUNTIME_ENV_ROOT`

Build entrypoint:
- `scripts/build_runtime_env.py --env-root <PFS path>`

Rule:
- Set `PFS_RUNTIME_ENV_ROOT` and materialize it with `scripts/build_runtime_env.py`.

## 3) Host bootstrap

The API host should run:

```bash
<PFS_RUNTIME_ENV_ROOT>/host-venv/bin/python scripts/run_server.py
```

`scripts/run_server.py` bootstraps the rest of the Python path from `PFS_RUNTIME_ENV_ROOT`, `PFS_TINKER_PATH`, and `PFS_HF_MODULES_PATH`.

There is no alternate per-package overlay mode anymore. Deployments must set
`PFS_RUNTIME_ENV_ROOT`.
