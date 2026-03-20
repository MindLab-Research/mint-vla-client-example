# Ops Console

新的 ops 前后端分离版本放在 `ops/`：

- `ops/backend/`: Python API，直接连接 mint API 和 Ray
- `ops/frontend/`: `pnpm + Vite + HeroUI` 的 deploy 界面

这版 deploy 不再依赖：

- `run_server.py` 进程扫描
- mint `/api/healthz` 轮询

旧的 `scripts/ops/` CLI/UI 已移除，后续统一维护 `ops/` 这一套前后端分离实现。

## 运行前提

后端需要运行在一个**同时能访问 mint API 和 Ray 集群**的位置。

你需要显式提供：

- `mint_base_url`
- `ray_address`（如果部署在 Ray driver 节点，默认可直接用 `auto`）
- `api_key`（如果目标 mint 开了鉴权）

## 本地开发

后端：

```bash
cd /vePFS-Mindverse/user/intern/nolanho/code/mint
uv run python -m ops.backend \
  --backend-port 8787 \
  --mint-base-url http://127.0.0.1:18000
```

也可以走环境变量：

```bash
export MINT_OPS_MINT_BASE_URL=http://127.0.0.1:18000
export MINT_OPS_API_KEY=your-key
uv run python -m ops.backend --backend-port 8787
```

如果不是部署在 driver 节点，再显式指定：

```bash
uv run python -m ops.backend \
  --backend-port 8787 \
  --mint-base-url http://127.0.0.1:18000 \
  --ray-address 127.0.0.1:6379
```

前端：

```bash
cd /vePFS-Mindverse/user/intern/nolanho/code/mint/ops/frontend
pnpm install
pnpm dev
```

默认开发地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8787`

前端会把 `/api/*` 代理到后端。

## 当前 deploy API

- `GET /api/deploy/state`
  - 直接请求 mint `/api/v1/actors`
  - 直接请求 Ray nodes / placement groups / actors
  - 不请求 mint `/api/healthz`
- `POST /api/deploy/actors/recycle`
  - 直接请求 mint `/api/v1/actors/kill`
- `POST /api/deploy/actors/rebuild`
  - 直接请求 mint create session / create model / sampling session 路径

## 构建前端

```bash
cd /vePFS-Mindverse/user/intern/nolanho/code/mint/ops/frontend
pnpm build
```

构建产物会输出到 `ops/frontend/dist/`。如果该目录存在，`ops.backend` 会自动把它作为静态页面服务出去。
