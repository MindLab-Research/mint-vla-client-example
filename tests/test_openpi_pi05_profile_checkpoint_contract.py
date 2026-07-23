from pathlib import Path

from mint_server.backend.openpi.openpi_pi05_worker import OpenPIPi05WorkerSession
from mint_server.backend.openpi.pi05_profiles import PI05_ACTION_LORA_R16_V1, write_profile_manifest


def test_profiled_sampler_export_completeness_requires_matching_manifest(tmp_path: Path) -> None:
    (tmp_path / "params").mkdir()
    (tmp_path / "assets").mkdir()

    assert not OpenPIPi05WorkerSession._sampler_export_complete(tmp_path, PI05_ACTION_LORA_R16_V1)
    assert OpenPIPi05WorkerSession._sampler_export_complete(tmp_path, None)

    write_profile_manifest(tmp_path, PI05_ACTION_LORA_R16_V1)
    assert OpenPIPi05WorkerSession._sampler_export_complete(tmp_path, PI05_ACTION_LORA_R16_V1)
