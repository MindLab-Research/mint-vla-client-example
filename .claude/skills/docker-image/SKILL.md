---
name: docker-image
description: |
  Build and publish Mint Docker images.

  Use for: updating Dockerfile deps, building `mint:N` locally, tagging to both registries, and pushing.

  Triggers: "docker image", "mint image", "build image", "push image", "mint:7", "mint:8", "mint:9"
---

# Docker Image (Mint) SOP

Goal: change `Dockerfile`, build a new `mint:N` locally (bump by 1), then publish to both registries.

## 1) Update Dockerfile and build locally

1. Edit `Dockerfile`.
2. Decide next version `mint:N` (increment from the most recent published tag).
3. Build locally:

```bash
# BuildKit may be unavailable (missing/broken buildx). Use legacy builder if needed.
DOCKER_BUILDKIT=0 docker build -t mint:N -f Dockerfile .
```

## 2) Tag image for both volcano and aliyun remotes

Example commands (replace tag number as needed):

```bash
docker tag mint:N acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:N
docker tag mint:N image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:N
```

## 3) Push image to both remotes

Example commands (replace tag number as needed):

```bash
docker push acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:N
docker push image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/mint:N
```

## 4) Update dev/prod Ray task YAMLs to the new mint:N

Update the image tag in:
- `.claude/skills/volcano-cluster/configs/mint-dev-head.yaml`
- `.claude/skills/volcano-cluster/configs/mint-dev-worker.yaml`
- `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`
- `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml`
