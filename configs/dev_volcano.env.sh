#!/bin/sh

# Compatibility wrapper. Runtime deployment configuration is owned outside the
# code repository under /share/mint/dev/config.

mint_dev_config_env="${MINT_DEV_CONFIG_ENV:-/share/mint/dev/config/common.env}"
if [ ! -r "${mint_dev_config_env}" ]; then
  echo "Missing dev config: ${mint_dev_config_env}" >&2
  return 1 2>/dev/null || exit 1
fi

. "${mint_dev_config_env}"
