"""orbax-checkpoint version-compat shim for openpi weight restoration.

openpi (vendored under runtime/.../src/openpi) pins
``orbax-checkpoint==0.11.13``, where ``PyTreeCheckpointer.metadata()``
returns a subscriptable mapping and ``openpi.models.model.restore_params``
does ``metadata["params"]``. The mint runtime ships orbax 0.11.40, where
``metadata()`` returns a ``StepMetadata`` object that is *not*
subscriptable; the ``{"params": ...}`` tree now lives on
``StepMetadata.item_metadata``.

Rather than patch the vendored upstream source, the openpi workers call
``install_restore_params_compat(openpi_model)`` once after import. The
patch is idempotent and a no-op on orbax versions where the original
``restore_params`` already works.
"""

from __future__ import annotations

import pathlib
from typing import Any

_PATCH_FLAG = "_mint_restore_params_orbax_compat"


def _restore_params_item_metadata(
    ocp: Any,
    jax: Any,
    traverse_util: Any,
    params_path: Any,
    *,
    restore_type: Any = None,
    dtype: Any = None,
    sharding: Any = None,
) -> Any:
    """orbax 0.11.40-compatible reimplementation of restore_params.

    Mirrors the upstream body but reads the params tree from
    ``StepMetadata.item_metadata`` (orbax >=0.11.3x) instead of the old
    subscriptable ``metadata["params"]``.
    """
    if restore_type is None:
        restore_type = jax.Array
    params_path = (
        pathlib.Path(params_path).resolve()
        if not str(params_path).startswith("gs://")
        else params_path
    )
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        tree_metadata = getattr(metadata, "item_metadata", metadata)
        item = {"params": tree_metadata["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=restore_type, dtype=dtype
                    ),
                    item,
                ),
            ),
        )["params"]

    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


def install_restore_params_compat(openpi_model: Any) -> None:
    """Idempotently wrap ``openpi_model.restore_params`` for orbax 0.11.40."""
    if getattr(openpi_model, _PATCH_FLAG, False):
        return

    import jax
    from flax import traverse_util
    import orbax.checkpoint as ocp

    _orig_restore_params = openpi_model.restore_params

    def _restore_params_compat(*args: Any, **kwargs: Any) -> Any:
        try:
            return _orig_restore_params(*args, **kwargs)
        except TypeError as exc:
            if "object is not subscriptable" not in str(exc):
                raise
            return _restore_params_item_metadata(
                ocp, jax, traverse_util, *args, **kwargs
            )

    openpi_model.restore_params = _restore_params_compat
    setattr(openpi_model, _PATCH_FLAG, True)
