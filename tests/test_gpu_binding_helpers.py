import os
from unittest.mock import patch

from mint_server.backend import gpu_binding_helpers as h


def test_physical_gpu_from_ray_id_maps_visible_index_to_physical_index():
    assert h._physical_gpu_from_ray_id("0", ["4", "7"]) == (4, None)
    assert h._physical_gpu_from_ray_id("1", ["4", "7"]) == (7, None)


def test_physical_gpu_from_ray_id_maps_visible_index_to_uuid():
    assert h._physical_gpu_from_ray_id("0", ["GPU-a", "GPU-b"]) == (None, "GPU-a")
    assert h._physical_gpu_from_ray_id("1", ["GPU-a", "GPU-b"]) == (None, "GPU-b")


def test_physical_gpu_from_ray_id_preserves_physical_index_without_visible_devices():
    assert h._physical_gpu_from_ray_id("3", []) == (3, None)
