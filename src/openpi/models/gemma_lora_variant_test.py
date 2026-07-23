import numpy as np
import pytest

from openpi.models import gemma
from openpi.training import weight_loaders


def test_gemma_300m_lora_r16_is_distinct_from_released_rank_32_variant() -> None:
    released = gemma.get_config("gemma_300m_lora")
    scheme_3 = gemma.get_config("gemma_300m_lora_r16")

    assert {name: config.rank for name, config in released.lora_configs.items()} == {"attn": 32, "ffn": 32}
    assert {name: config.rank for name, config in scheme_3.lora_configs.items()} == {"attn": 16, "ffn": 16}
    assert {name: config.alpha for name, config in scheme_3.lora_configs.items()} == {"attn": 16.0, "ffn": 16.0}


def test_checkpoint_merge_initializes_only_missing_lora_tensors() -> None:
    reference = {
        "base": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "adapter": {"lora_a": np.zeros((2, 16), dtype=np.float32)},
    }
    loaded = {"base": {"kernel": np.full((2, 2), 3, dtype=np.float32)}}

    merged = weight_loaders._merge_params(loaded, reference, missing_regex=".*lora.*")

    np.testing.assert_array_equal(merged["base"]["kernel"], loaded["base"]["kernel"])
    np.testing.assert_array_equal(merged["adapter"]["lora_a"], reference["adapter"]["lora_a"])
    with pytest.raises(KeyError):
        weight_loaders._merge_params({}, reference, missing_regex=".*lora.*")["base"]
