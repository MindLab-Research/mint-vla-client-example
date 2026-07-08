#!/usr/bin/env bash
# step4_health.sh — 本机化健康检查（替代三机版 step4.sh 的 ssh driver curl）
# server 就在本机 :30496，直接 curl。
set -uo pipefail
PORT="${MINT_PORT:-30496}"
echo "=== healthz（期望 {\"status\":\"ready\"}）==="
curl -s "http://localhost:${PORT}/api/v1/healthz"; echo
echo "=== server_info ==="
curl -s "http://localhost:${PORT}/api/v1/server_info"; echo
echo "=== admission_stats（scheduler + supervisor + ray cluster）==="
curl -s "http://localhost:${PORT}/internal/admission_stats"; echo
