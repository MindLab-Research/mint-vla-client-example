---
name: docker-image
description: |
  Build and publish Mint Docker images.

  Use for: updating Dockerfile deps, building `mint:N` locally, tagging to both registries, and pushing.

  Triggers: "docker image", "mint image", "build image", "push image", "mint:7", "mint:8", "mint:9"
---

# Docker Image (Mint) SOP

Goal: change `Dockerfile`, build `mint:16-*` locally, then publish to the private registries.

We maintain 2 CUDA-arch variants starting at version 16:
- `mint:16-sm80`: Volcano A-cards (A800, SM80)
- `mint:16-sm90`: Aliyun H-cards (H, SM90)

Why two variants:
- DeepEP upstream is SM90-centric. `mint:16-sm80` rebuilds DeepEP with SM90 features disabled so it runs on SM80 GPUs.
- `mint:16-sm90` uses the base-image DeepEP build (no DeepEP rebuild in our Dockerfile).

## 1) Update Dockerfile and build locally

1. Edit `Dockerfile`.
2. Decide which variant(s) to build (`sm80`, `sm90`).
3. Build locally (start from version 16):

```bash
# BuildKit may be unavailable (missing/broken buildx). Use legacy builder if needed.

# Volcano (A800 / SM80): rebuild DeepEP inside the image
DOCKER_BUILDKIT=0 docker build \
  --build-arg DEEPEP_VARIANT=sm80 \
  -t mint:16-sm80 \
  -f Dockerfile .

# Aliyun (H / SM90): use DeepEP from the base image
DOCKER_BUILDKIT=0 docker build \
  --build-arg DEEPEP_VARIANT=base \
  -t mint:16-sm90 \
  -f Dockerfile .
```

## 2) Tag image for both volcano and aliyun remotes

Example commands:

```bash
docker tag mint:16-sm80 acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:16-sm80
docker tag mint:16-sm80 image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:16-sm80

docker tag mint:16-sm90 acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:16-sm90
docker tag mint:16-sm90 image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:16-sm90
```

## 3) Push image to both remotes

Example commands:

```bash
docker push acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:16-sm80
docker push image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:16-sm80

docker push acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:16-sm90
docker push image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:16-sm90
```

## 4) Update dev/prod Ray task YAMLs to the new mint:N

Update the image tag in:
- `.claude/skills/volcano-cluster/configs/mint-dev-head.yaml`
- `.claude/skills/volcano-cluster/configs/mint-dev-worker.yaml`
- `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`
- `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml`

Guideline:
- Volcano should use `mint:16-sm80`
- Aliyun should use `mint:16-sm90`
