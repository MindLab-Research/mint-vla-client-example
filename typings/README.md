# Typing stubs for GPU-only and CPU-ininstallable dependencies

These stubs allow pyright to resolve modules that cannot be installed in
the CI environment (GPU-only packages, CUDA extensions, etc.).

Each stub exports `Any` for all members, which is intentional — the goal
is import resolution, not type accuracy for these external packages.
