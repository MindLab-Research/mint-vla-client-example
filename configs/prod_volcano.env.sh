#!/bin/sh

# Compatibility wrapper. Runtime deployment configuration is owned outside the
# code repository under /vePFS-Mindverse/share/mint/prod/config.

mint_prod_config_env="${MINT_PROD_CONFIG_ENV:-/vePFS-Mindverse/share/mint/prod/config/prod.env}"
if [ ! -r "${mint_prod_config_env}" ]; then
  echo "Missing prod config: ${mint_prod_config_env}" >&2
  return 1 2>/dev/null || exit 1
fi

. "${mint_prod_config_env}"
