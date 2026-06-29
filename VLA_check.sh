#!/usr/bin/env bash
# VLA (openpi) end-to-end smoke check.
#
# Unlike RLcheck.sh (token-modality RL), openpi models are ACTION-modality VLA
# policies. They use a different code path on the server:
#   create_model -> /api/v1/mint/vla/train_step -> save_weights_for_sampler
#   -> action_sessions/act
# The driver script scripts/wip/openpi_vla_smoke.py exercises that full path
# with synthetic data (1x1 PNG images + fake observations), so no external
# LIBERO dataset is needed.
#
# PREREQUISITES (see notes at bottom):
#   1. The openpi model MUST be in the server's MINT_SUPPORTED_MODELS, otherwise
#      create_model is rejected. Add it in step3.sh and restart the server.
#   2. The server must have openpi checkpoint paths configured
#      (MINT_OPENPI_FAST_* / MINT_OPENPI_FAST_CHECKPOINT_BASE_DIR), otherwise
#      training will fail to locate weights.

set -u

# Reuse the SSH tunnel to the dev driver (ignore "address already in use" if a
# tunnel from RLcheck.sh is still up).
ssh -f -N -L 8000:localhost:30496 driver 2>/dev/null || true

MINT_CODE_ROOT=/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi

# pi0-fast (AR action tokens, cross_entropy) is the default; pass pi05 as $1 to
# test the flow-matching variant instead.
MODEL="${1:-openpi/pi0-fast-libero-low-mem-finetune}"

MINT_BASE_URL=http://localhost:8000 \
MINT_API_KEY=tml-dummy \
python "${MINT_CODE_ROOT}/scripts/wip/openpi_vla_smoke.py" \
  --model "${MODEL}" \
  --output-json /tmp/vla_check_result.json
