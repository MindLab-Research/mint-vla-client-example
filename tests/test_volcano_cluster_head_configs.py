import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
VOLCANO_ROOT = REPO_ROOT / ".claude" / "skills" / "volcano-cluster"
DEV_HEAD = VOLCANO_ROOT / "configs" / "mint-dev-head.yaml"
DEV_WORKER = VOLCANO_ROOT / "configs" / "mint-dev-worker.yaml"
PROD_HEAD = VOLCANO_ROOT / "configs" / "mint-prod-head.yaml"
PROD_WORKER = VOLCANO_ROOT / "configs" / "mint-prod-worker.yaml"
MINT_RAY_NODE = VOLCANO_ROOT / "runtime" / "supervisor" / "current" / "bin" / "mint-ray-node"


def _load_mint_ray_node(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_ray.util = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
    fake_ray_private = types.ModuleType("ray._private")
    fake_node = types.ModuleType("ray._private.node")
    fake_parameter = types.ModuleType("ray._private.parameter")

    class _FakeNode:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def dead_processes(self):
            return []

        def kill_all_processes(self, **_kwargs) -> None:
            pass

    class _FakeRayParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_node.Node = _FakeNode
    fake_parameter.RayParams = _FakeRayParams
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray._private", fake_ray_private)
    monkeypatch.setitem(sys.modules, "ray._private.node", fake_node)
    monkeypatch.setitem(sys.modules, "ray._private.parameter", fake_parameter)

    loader = importlib.machinery.SourceFileLoader("mint_ray_node_under_test", str(MINT_RAY_NODE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _config_texts() -> dict[str, str]:
    return {
        "dev-head": DEV_HEAD.read_text(encoding="utf-8"),
        "dev-worker": DEV_WORKER.read_text(encoding="utf-8"),
        "prod-head": PROD_HEAD.read_text(encoding="utf-8"),
        "prod-worker": PROD_WORKER.read_text(encoding="utf-8"),
    }


def test_volcano_templates_mount_vepfs_tmp_root() -> None:
    for name, text in _config_texts().items():
        assert "Ray tmp uses /mnt/tmp" in text, name
        assert "export MINT_TMP_ROOT=/mnt/tmp" in text, name
        assert 'Type: "Vepfs"' in text, name
        assert 'MountPath: "/mnt/tmp"' in text, name
        assert 'ReadOnly: false' in text, name


def test_volcano_templates_delegate_ray_start_to_shared_runtime() -> None:
    for name, text in _config_texts().items():
        assert "MINT_RUNTIME_ROOT=/vePFS-Mindverse/share/mint/runtime" in text, name
        assert "MINT_NODE_SERVICE_DIR=" in text, name
        assert "runsvdir" in text, name
        assert "object_spilling_directory=" not in text, name
        assert "RAY_TMP_LINK" not in text, name
        assert "/tmp/md" not in text, name
        assert "/tmp/mp" not in text, name


def test_mint_ray_node_uses_tmp_root_for_ray_paths(monkeypatch, tmp_path) -> None:
    module = _load_mint_ray_node(monkeypatch)
    monkeypatch.setenv("MINT_TMP_ROOT", str(tmp_path))

    temp_dir, object_spilling_dir, os_tmpdir, xdg_cache_home = module._prepare_tmp("worker", "node-a")

    assert temp_dir == str(tmp_path / "w" / "node-a" / "t")
    assert object_spilling_dir == str(tmp_path / "w" / "node-a" / "s")
    assert os_tmpdir == str(tmp_path / "w" / "node-a" / "tmp")
    assert xdg_cache_home == str(tmp_path / "w" / "node-a" / "cache")
    assert module.os.environ["TMPDIR"] == os_tmpdir
    assert module.os.environ["XDG_CACHE_HOME"] == xdg_cache_home


def test_mint_ray_node_mountinfo_uses_longest_mount_prefix(monkeypatch) -> None:
    module = _load_mint_ray_node(monkeypatch)

    entries = [
        {"mount_point": "/", "fs_type": "overlay", "mount_source": "overlay"},
        {"mount_point": "/mnt", "fs_type": "tmpfs", "mount_source": "tmpfs"},
        {"mount_point": "/mnt/tmp", "fs_type": "vepfs", "mount_source": "vepfs-cnbjecc87dad63ea"},
    ]

    result = module._mount_for_path(Path("/mnt/tmp/w/node-a/s"), entries)

    assert result == entries[2]


def test_mint_ray_node_decodes_mountinfo_escaped_paths(monkeypatch, tmp_path) -> None:
    module = _load_mint_ray_node(monkeypatch)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "23 1 0:22 / /mnt/tmp rw,relatime - vepfs vepfs-cnbjecc87dad63ea rw\n"
        "24 1 0:23 / /mnt/tmp/foo\\040bar rw,relatime - fuse.vepfs source\\040name rw\n",
        encoding="utf-8",
    )

    entries = module._read_mountinfo(str(mountinfo))

    assert entries == [
        {"mount_point": "/mnt/tmp", "fs_type": "vepfs", "mount_source": "vepfs-cnbjecc87dad63ea"},
        {"mount_point": "/mnt/tmp/foo bar", "fs_type": "fuse.vepfs", "mount_source": "source name"},
    ]


def test_mint_ray_node_low_space_is_warning_not_failure(monkeypatch, tmp_path) -> None:
    module = _load_mint_ray_node(monkeypatch)

    monkeypatch.setattr(
        module.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=1, f_frsize=1024, f_blocks=10),
    )

    record = module._path_diagnostic(
        role="worker",
        node_id="node-a",
        kind="object_spilling_directory",
        path=str(tmp_path),
        min_free_bytes=2048,
        mountinfo=[
            {
                "mount_point": str(tmp_path),
                "fs_type": "vepfs",
                "mount_source": "vepfs-cnbjecc87dad63ea",
            }
        ],
    )

    assert record["warnings"] == ["low_free_space"]
    assert record["free_bytes"] == 1024
    assert record["total_bytes"] == 10240
    assert record["writable"] is True
    assert record["fs_type"] == "vepfs"


def test_mint_ray_node_logs_structured_path_checks(monkeypatch, tmp_path, capsys) -> None:
    module = _load_mint_ray_node(monkeypatch)
    monkeypatch.setenv("MINT_RAY_WARN_TMP_FREE_BYTES", "2048")
    monkeypatch.setattr(
        module.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=1, f_frsize=1024, f_blocks=10),
    )
    monkeypatch.setattr(
        module,
        "_read_mountinfo",
        lambda: [{"mount_point": str(tmp_path), "fs_type": "vepfs", "mount_source": "vepfs-cnbjecc87dad63ea"}],
    )

    module._log_path_diagnostics("worker", "node-a", {"object_spilling_directory": str(tmp_path)})

    out = capsys.readouterr().out.strip()
    assert out.startswith("mint ray path check ")
    payload = json.loads(out.removeprefix("mint ray path check "))
    assert payload["kind"] == "object_spilling_directory"
    assert payload["warnings"] == ["low_free_space"]
    assert payload["fs_type"] == "vepfs"
