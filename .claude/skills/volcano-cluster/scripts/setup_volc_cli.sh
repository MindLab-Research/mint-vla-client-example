#!/bin/bash
# Setup Volcano ML Platform CLI
# Usage: ./setup_volc_cli.sh

set -e

echo "Installing Volcano CLI..."
sh -c "$(curl -fsSL https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/cli-binary/install.sh)"

# Add to PATH
VOLC_BIN="$HOME/.volc/bin"
if [[ ":$PATH:" != *":$VOLC_BIN:"* ]]; then
    echo "export PATH=\$HOME/.volc/bin:\$PATH" >> ~/.bashrc
    export PATH="$VOLC_BIN:$PATH"
fi

echo ""
echo "CLI installed. Run 'volc configure' to set up credentials."
echo ""
echo "You'll need:"
echo "  - AK (Access Key)"
echo "  - SK (Secret Key)"
echo "  - Region (cn-beijing)"
