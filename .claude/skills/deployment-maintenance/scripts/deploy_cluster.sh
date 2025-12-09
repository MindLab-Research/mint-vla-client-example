#!/bin/bash
# Deploy Ray cluster to Volcano ML Platform
# Usage: ./deploy_cluster.sh [simple|scalable]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"
MODE="${1:-simple}"

case "$MODE" in
    simple)
        echo "Deploying single-node Ray cluster (8 GPUs)..."
        volc ml_task submit -c "$CONFIG_DIR/ray_cluster_8gpu.yaml"
        echo ""
        echo "Cluster submitted. Check status with: volc ml_task list"
        ;;
    scalable)
        echo "Deploying scalable Ray cluster (separate head + workers)..."
        echo ""
        echo "Step 1: Starting head node..."
        volc ml_task submit -c "$CONFIG_DIR/ray_master.yaml"
        echo ""
        echo "Head node submitted."
        echo ""
        echo "Next steps:"
        echo "  1. Wait for head node to start: volc ml_task list"
        echo "  2. Get head node IP from task details"
        echo "  3. Update ray_worker_8gpu.yaml with HEAD_IP"
        echo "  4. Submit worker: volc ml_task submit -c $CONFIG_DIR/ray_worker_8gpu.yaml"
        ;;
    *)
        echo "Usage: $0 [simple|scalable]"
        echo ""
        echo "  simple    - Single node with head + 8 GPUs (default)"
        echo "  scalable  - Separate head node + GPU workers"
        exit 1
        ;;
esac
