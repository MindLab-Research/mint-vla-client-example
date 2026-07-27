# Parent workspace legacy files

The local ignored directory `archive/parent-vla-legacy/` contains files that were loose siblings of the repository before the workspace was consolidated on 2026-07-23. They are preserved only to avoid losing historical content and are not maintained entry points.

Canonical replacements:

| Archived file | Maintained file |
|---|---|
| `Tutorial.md` | `Tutorial.md` |

The former loose `inspect_mano_lance.py` was byte-identical to `scripts/tools/inspect_mano_lance.py` and was deduplicated rather than archived.

The former parent-level `results/` snapshot is retained locally at `results/legacy-parent-vla/`; generated results are Git-ignored.

Historical MANO inference scripts were removed after Mode4 became the sole maintained evaluator at `scripts/eval/infer_mano_mode4.py`.
