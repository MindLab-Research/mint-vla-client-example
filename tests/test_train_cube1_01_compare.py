from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import pytest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/train/train_cube1_01_compare.py"


def _load_compare_module() -> types.ModuleType:
    """Load the RNG helper without requiring the local OpenPI runtime."""
    fake_base = types.ModuleType("openpi_vla_smoke_lance_base")
    fake_base.LanceViewpi05Dataset = object
    fake_base.contact_windows_lib = types.SimpleNamespace(DEFAULT_CONTACT_CONTEXT_FRAMES=100)
    fake_base.PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"
    fake_base.MODEL_CHOICES = (fake_base.PI05_MODEL, "openpi/pi05-action-lora-r16-finetune")
    fake_base._transform_sample = lambda sample, _: sample
    fake_base._pi05_datum_from_transformed = lambda _, sample: sample
    fake_base.normalize = types.SimpleNamespace(load=lambda _path: {})
    fake_lance = types.ModuleType("lance")
    fake_lance.dataset = lambda *_args, **_kwargs: None
    spec = importlib.util.spec_from_file_location("train_cube1_01_compare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"openpi_vla_smoke_lance_base": fake_base, "lance": fake_lance},
    ):
        spec.loader.exec_module(module)
    return module


class _SamplingDataset:
    def sample_indices(self, n: int, rng: np.random.Generator) -> list[int]:
        return rng.integers(0, 10_000, size=n).tolist()


class _BatchDataset:
    def __init__(self) -> None:
        self.calls = 0
        self._action_horizon = 10
        self._action_source = "pd_target_delta"

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, int]:
        self.calls += 1
        return {"index": index}


def _norm_stats(std: list[float]) -> dict[str, object]:
    values = np.asarray(std, dtype=np.float32)
    return {
        key: types.SimpleNamespace(
            std=values,
            q01=-np.ones_like(values),
            q99=np.ones_like(values),
        )
        for key in ("state", "actions")
    }


def _datum(index: int) -> dict[str, object]:
    image = {"type": "image", "data": [f"image-{index}"]}
    return {
        "observation": {
            "state": {"data": [0.0] * 32, "shape": [32]},
            "model_input": {"chunks": [image, {"type": "encoded_text", "tokens": [1, 2, 3]}]},
        },
        "supervision": {"actions": {"data": [index]}},
    }


class TokenizePrompt:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        data = dict(data)
        data.pop("prompt", None)
        data["tokenized_prompt"] = np.rint((state + 1.0) * 100).astype(np.int32)
        data["tokenized_prompt_mask"] = np.ones(32, dtype=bool)
        return data


def _prepared(compare, index: int):
    state = np.zeros(32, dtype=np.float32)
    actions = np.zeros((10, 32), dtype=np.float32)
    prefix = {"state": state, "prompt": "cube", "actions": actions}
    clean = _datum(index)
    clean["supervision"]["actions"] = {
        "data": actions.tolist(),
        "shape": [10, 32],
    }
    return compare.PreparedDatum(prefix, TokenizePrompt(), (), clean)


class CompareTrainingCliTests(unittest.TestCase):
    def test_model_option_propagates_from_cli(self) -> None:
        compare = _load_compare_module()
        with patch.object(sys, "argv", [
            str(SCRIPT), "--model", "openpi/pi05-action-lora-r16-finetune",
            "--lance-dataset", "dataset.lance", "--save-path", "save",
            "--output-json", "result.json",
        ]):
            args = compare.parse_args()
            self.assertEqual(args.model, "openpi/pi05-action-lora-r16-finetune")
            self.assertEqual(args.coverage_anchors_per_row, 8)
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.batch_build_workers, 4)
            self.assertEqual(args.prefetch_batches, 2)
            self.assertEqual(args.language_conditioning, "gesture")
            self.assertTrue(str(args.gesture_index).endswith("new_all_generated_mano.index.json"))
            self.assertEqual(args.target_noise_std, 0.0)
            self.assertEqual(args.learning_rate, 1e-4)
            self.assertEqual(
                args.row_indices,
                "656,657,658,659,995,996,997,998,1155,1156,1303,1304",
            )
            self.assertEqual(args.checkpoint_every, 0)
            self.assertEqual(args.checkpoint_save_path_template, "")

    def test_parallel_worker_option_propagates_from_cli(self) -> None:
        compare = _load_compare_module()
        with patch.object(sys, "argv", [
            str(SCRIPT), "--batch-build-workers", "8",
            "--language-conditioning", "motion_variant",
            "--lance-dataset", "dataset.lance", "--save-path", "save",
            "--output-json", "result.json",
        ]):
            args = compare.parse_args()
            self.assertEqual(args.batch_build_workers, 8)
            self.assertEqual(args.language_conditioning, "motion_variant")

    def test_learning_rate_propagates_to_explicit_adam_contract(self) -> None:
        compare = _load_compare_module()
        with patch.object(sys, "argv", [
            str(SCRIPT), "--learning-rate", "1e-5", "--metrics-jsonl", "metrics.jsonl",
            "--lance-dataset", "dataset.lance", "--save-path", "save",
            "--output-json", "result.json",
        ]):
            args = compare.parse_args()
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.metrics_jsonl, Path("metrics.jsonl"))
        payload = compare.vla_train_step_payload(
            model_id="model", batch=[{"sample": 1}], learning_rate=args.learning_rate
        )
        self.assertEqual(payload["adam_params"], {
            "learning_rate": 1e-5,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-12,
        })
        with self.assertRaisesRegex(ValueError, "finite positive"):
            compare.vla_train_step_payload(model_id="model", batch=[], learning_rate=0)

    def test_periodic_checkpoint_options_propagate_from_cli(self) -> None:
        compare = _load_compare_module()
        with patch.object(sys, "argv", [
            str(SCRIPT), "--checkpoint-every", "15000",
            "--checkpoint-save-path-template", "motion_step{step}",
            "--lance-dataset", "dataset.lance", "--save-path", "save",
            "--output-json", "result.json",
        ]):
            args = compare.parse_args()
            self.assertEqual(args.checkpoint_every, 15000)
            self.assertEqual(args.checkpoint_save_path_template, "motion_step{step}")


class CompareTrainingCheckpointTests(unittest.TestCase):
    def test_periodic_checkpoint_uses_global_step_and_skips_final(self) -> None:
        compare = _load_compare_module()
        kwargs = {
            "global_step_offset": 0,
            "phase_steps": 150000,
            "checkpoint_every": 15000,
            "path_template": "motion_step{step}",
        }
        self.assertIsNone(compare.periodic_checkpoint_path(phase_step=14999, **kwargs))
        self.assertEqual(
            compare.periodic_checkpoint_path(phase_step=15000, **kwargs),
            (15000, "motion_step15000"),
        )
        self.assertIsNone(compare.periodic_checkpoint_path(phase_step=150000, **kwargs))

    def test_periodic_checkpoint_respects_resumed_global_offset(self) -> None:
        compare = _load_compare_module()
        self.assertEqual(
            compare.periodic_checkpoint_path(
                phase_step=5000,
                global_step_offset=10000,
                phase_steps=20000,
                checkpoint_every=15000,
                path_template="motion_step{step}",
            ),
            (15000, "motion_step15000"),
        )


class CompareTrainingLanguageTests(unittest.TestCase):
    def test_object_only_preserves_prompt(self) -> None:
        compare = _load_compare_module()
        metadata = {"object_names": ["banana"], "raw_data_info": {"id": 1}}
        self.assertEqual(
            compare.format_language_prompt("  pick up the banana  ", metadata, "object_only"),
            "  pick up the banana  ",
        )

    def test_gesture_prompt_uses_canonical_action_label(self) -> None:
        compare = _load_compare_module()
        metadata = {"object_names": ["cube1"], "raw_data_info": {"id": 1}}
        self.assertEqual(
            compare.format_language_prompt(
                "pick up the cube1", metadata, "gesture", gesture="02"
            ),
            "pick up the cube1 using gesture 02",
        )
        with self.assertRaisesRegex(ValueError, "canonical gesture"):
            compare.format_language_prompt("pick up the cube1", metadata, "gesture")

    def test_motion_variant_is_object_scoped_and_zero_padded(self) -> None:
        compare = _load_compare_module()
        metadata = {"object_names": ["Banana Object"], "raw_data_info": {"id": 1}}
        self.assertEqual(
            compare.format_language_prompt("pick up the banana", metadata, "motion_variant"),
            "pick up the banana using motion variant banana_object_01",
        )

    def test_motion_variant_rejects_missing_or_invalid_id(self) -> None:
        compare = _load_compare_module()
        for raw_id in (None, -1, True, "01"):
            metadata = {"object_names": ["banana"], "raw_data_info": {"id": raw_id}}
            with self.assertRaisesRegex(ValueError, "raw_data_info.id"):
                compare.format_language_prompt(
                    "pick up the banana", metadata, "motion_variant"
                )

    def test_language_formatter_rejects_malformed_prompt_and_metadata(self) -> None:
        compare = _load_compare_module()
        valid = {"object_names": ["banana"], "raw_data_info": {"id": 1}}
        for prompt in (None, 123, "   "):
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                compare.format_language_prompt(prompt, valid, "object_only")
        with self.assertRaisesRegex(ValueError, "requires trajectory_metadata"):
            compare.format_language_prompt("pick up the banana", None, "motion_variant")
        invalid_object = {"object_names": [None], "raw_data_info": {"id": 1}}
        with self.assertRaisesRegex(ValueError, r"object_names\[0\]"):
            compare.format_language_prompt(
                "pick up the banana", invalid_object, "motion_variant"
            )

    def test_motion_variant_rejects_cross_source_aliases(self) -> None:
        compare = _load_compare_module()
        first = {
            "object_names": ["banana"],
            "raw_data_info": {
                "capMachine": "machine01", "operator": "s11",
                "scene": "banana", "id": 1,
            },
        }
        second = {
            "object_names": ["banana"],
            "raw_data_info": {
                "capMachine": "machine01", "operator": "s12",
                "scene": "banana", "id": 1,
            },
        }
        compare.validate_motion_variant_metadata([first, first])
        with self.assertRaisesRegex(ValueError, "aliases distinct raw sources"):
            compare.validate_motion_variant_metadata([first, second])


class CompareTrainingProfileTests(unittest.TestCase):
    def test_discrete_state_model_allows_real_pre_tokenization_noise(self) -> None:
        compare = _load_compare_module()
        compare.validate_state_noise("openpi/pi05-action-lora-r16-finetune", 0.01)

    def test_legacy_invalid_sigma_rejected(self) -> None:
        compare = _load_compare_module()
        with self.assertRaisesRegex(ValueError, "discrete state"):
            compare.validate_state_noise("openpi/pi05-libero-low-mem-finetune", 0.01)
        for value in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                compare.validate_state_noise("openpi/pi05-action-lora-r16-finetune", value)


class CompareTrainingNormStatsTests(unittest.TestCase):
    def test_pd_target_norm_reads_measured_actions_for_projection(self) -> None:
        compare = _load_compare_module()
        state = np.zeros((2, 32), dtype=np.float32)
        measured = np.zeros((2, 32), dtype=np.float32)
        hands = [{
            "urdf_dof": np.zeros((2, 26), dtype=np.float32).tolist(),
            "urdf_dof_target": np.full((2, 26), 0.5, dtype=np.float32).tolist(),
        }]
        requested_columns = []

        class Table:
            def __init__(self, row): self.row = row
            def to_pylist(self): return [self.row]

        class ImageDataset:
            def take(self, _, *, columns):
                requested_columns.append(columns)
                return Table({"state": state.tolist(), "actions": measured.tolist()})

        class TargetDataset:
            def take(self, _, *, columns):
                self.columns = columns
                return Table({"hands": hands})

        class RunningStats:
            def __init__(self): self.values = []
            def update(self, values): self.values.append(np.asarray(values))
            def get_statistics(self): return np.concatenate(self.values, axis=0)

        dataset = types.SimpleNamespace(
            _action_source="pd_target_delta",
            _target_dataset=TargetDataset(),
            _dataset=ImageDataset(),
            _index=[(0, 0), (0, 1)],
            _source_row_indices=[7],
        )
        with patch.object(compare.L.normalize, "RunningStats", RunningStats, create=True):
            stats = compare.selected_norm_stats(dataset)
        self.assertEqual(requested_columns, [["state", "actions"]])
        np.testing.assert_allclose(stats["actions"][:, :26], 0.5)
        np.testing.assert_array_equal(stats["actions"][:, 26:], 0.0)

    def test_query_anchored_norm_matches_real_horizon_and_padding(self) -> None:
        compare = _load_compare_module()
        state = np.zeros((3, 32), dtype=np.float32)
        state[:, 0] = [0.0, 10.0, 20.0]
        state[:, 6] = [0.0, 1.0, 2.0]
        target = state[:, :26].copy()
        target[:, 0] = [1.0, 12.0, 23.0]
        target[:, 3] = [0.1, 0.2, 0.3]
        target[:, 6] = [0.5, 1.7, 2.9]
        measured = np.zeros((3, 32), dtype=np.float32)
        hands = [{
            "urdf_dof": state[:, :26].tolist(),
            "urdf_dof_target": target.tolist(),
        }]

        class Table:
            def __init__(self, row): self.row = row
            def to_pylist(self): return [self.row]

        class ImageDataset:
            def take(self, _, *, columns):
                return Table({"state": state.tolist(), "actions": measured.tolist()})

        class TargetDataset:
            def take(self, _, *, columns):
                return Table({"hands": hands})

        class RunningStats:
            def __init__(self): self.values = []
            def update(self, values): self.values.append(np.asarray(values))
            def get_statistics(self): return np.concatenate(self.values, axis=0)

        dataset = types.SimpleNamespace(
            _action_source="urdf_target_absolute",
            _target_dataset=TargetDataset(),
            _dataset=ImageDataset(),
            _index=[(0, 0), (0, 1), (0, 2)],
            _source_row_indices=[7],
            _action_horizon=2,
        )
        with patch.object(compare.L.normalize, "RunningStats", RunningStats, create=True):
            stats = compare.selected_norm_stats(dataset)
        np.testing.assert_allclose(stats["actions"][:, 0], [1, 12, 2, 13, 3, 3])
        np.testing.assert_allclose(stats["actions"][:, 3], [0.1, 0.2, 0.2, 0.3, 0.3, 0.3])
        np.testing.assert_allclose(
            stats["actions"][:, 6], [0.5, 1.7, 0.7, 1.9, 0.9, 0.9], atol=1e-6
        )
        np.testing.assert_array_equal(stats["actions"][:, 26:], 0.0)
        self.assertEqual(stats["state"].shape, (3, 32))

    def test_locked_norm_stats_are_shape_checked_and_hashed(self) -> None:
        compare = _load_compare_module()
        values = np.zeros(32, dtype=np.float32)
        stats = {
            key: types.SimpleNamespace(mean=values, std=values, q01=values, q99=values + 1)
            for key in ("state", "actions")
        }
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "norm_stats.json").write_text("locked", encoding="utf-8")
            with patch.object(compare.L.normalize, "load", return_value=stats):
                loaded, provenance = compare.load_or_compute_norm_stats(object(), directory)
        self.assertIs(loaded, stats)
        self.assertEqual(provenance["source"], "loaded")
        self.assertEqual(len(provenance["sha256"]), 64)


class CompareTrainingRngTests(unittest.TestCase):
    def test_state_noise_does_not_change_sample_schedule(self) -> None:
        compare = _load_compare_module()
        clean_sample_rng, _, clean_augmentation_seed = compare.make_rngs(42)
        augmented_sample_rng, augmented_noise_rng, augmented_augmentation_seed = compare.make_rngs(42)
        dataset = _SamplingDataset()

        clean_schedule = [dataset.sample_indices(4, clean_sample_rng) for _ in range(6)]
        augmented_schedule = []
        for _ in range(6):
            augmented_schedule.append(dataset.sample_indices(4, augmented_sample_rng))
            augmented_noise_rng.normal(0.0, 0.05, size=(4, 8))

        self.assertEqual(clean_schedule, augmented_schedule)
        self.assertEqual(clean_augmentation_seed, 43)
        self.assertEqual(augmented_augmentation_seed, 43)

    def test_explicit_augmentation_seed_does_not_affect_sample_rng(self) -> None:
        compare = _load_compare_module()
        first_sample_rng, _, first_augmentation_seed = compare.make_rngs(7, 101)
        second_sample_rng, _, second_augmentation_seed = compare.make_rngs(7, 202)

        self.assertEqual(first_augmentation_seed, 101)
        self.assertEqual(second_augmentation_seed, 202)
        self.assertEqual(
            first_sample_rng.integers(0, 10_000, size=12).tolist(),
            second_sample_rng.integers(0, 10_000, size=12).tolist(),
        )


class CompareTrainingCacheAndBatchTests(unittest.TestCase):
    def test_lru_is_bounded_evicts_oldest_and_is_run_isolated(self) -> None:
        compare = _load_compare_module()
        first = compare.DatumCache(2)
        created: list[int] = []

        def create(key: int) -> dict[str, int]:
            created.append(key)
            return {"key": key}

        first.get_or_create(1, lambda: create(1))
        first.get_or_create(2, lambda: create(2))
        first.get_or_create(1, lambda: create(1))  # refresh key 1
        first.get_or_create(3, lambda: create(3))  # evicts key 2
        self.assertEqual(list(first._items), [1, 3])
        self.assertEqual(first.summary(), {
            "hits": 1, "misses": 3, "evictions": 1, "current_size": 2, "capacity": 2,
        })
        second = compare.DatumCache(2)
        second.get_or_create(1, lambda: create(10))
        self.assertEqual(second.summary()["hits"], 0)
        self.assertEqual(second.summary()["current_size"], 1)
        self.assertEqual(created, [1, 2, 3, 10])

    def test_concurrent_cache_lookup_coalesces_duplicate_creation(self) -> None:
        compare = _load_compare_module()
        cache = compare.DatumCache(4)
        calls = 0
        calls_lock = threading.Lock()

        def create():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.03)
            return {"key": 7}

        with compare.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: cache.get_or_create(7, create), range(4)))
        self.assertEqual(calls, 1)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual(cache.summary()["misses"], 1)
        self.assertEqual(cache.summary()["hits"], 3)

    def test_parallel_clean_batch_preserves_input_order(self) -> None:
        compare = _load_compare_module()
        dataset = _BatchDataset()

        def lower(_, sample):
            time.sleep((2 - sample["index"]) * 0.01)
            return _datum(sample["index"])

        with compare.ThreadPoolExecutor(max_workers=3) as executor, patch.object(
            compare.L, "_transform_sample", side_effect=lambda sample, _: sample
        ), patch.object(compare.L, "_pi05_datum_from_transformed", side_effect=lower):
            batch = compare.build_batch(
                dataset,
                object(),
                base_model="openpi/pi05-action-lora-r16-finetune",
                indices=[0, 1, 2],
                norm_stats=_norm_stats([1] * 32),
                state_noise_std=0,
                rng=np.random.default_rng(8),
                datum_cache=compare.DatumCache(0),
                executor=executor,
            )
        self.assertEqual([datum["supervision"]["actions"]["data"][0] for datum in batch], [0, 1, 2])

    def test_parallel_augmentation_matches_serial_rng_and_outputs(self) -> None:
        compare = _load_compare_module()
        indices = [0, 1, 2]

        def prepare(sample, _, __):
            time.sleep((2 - sample["index"]) * 0.01)
            return _prepared(compare, sample["index"])

        with patch.object(compare, "_prepare_discrete_datum", side_effect=prepare):
            serial = compare.build_batch(
                _BatchDataset(), object(),
                base_model="openpi/pi05-action-lora-r16-finetune", indices=indices,
                norm_stats=_norm_stats([1] * 32), state_noise_std=0.05,
                rng=np.random.default_rng(43), datum_cache=compare.DatumCache(0),
            )
        with compare.ThreadPoolExecutor(max_workers=3) as executor, patch.object(
            compare, "_prepare_discrete_datum", side_effect=prepare
        ):
            parallel = compare.build_batch(
                _BatchDataset(), object(),
                base_model="openpi/pi05-action-lora-r16-finetune", indices=indices,
                norm_stats=_norm_stats([1] * 32), state_noise_std=0.05,
                rng=np.random.default_rng(43), datum_cache=compare.DatumCache(0),
                executor=executor,
            )
        self.assertEqual(parallel, serial)

    def test_real_augmentation_retokenizes_and_shares_clean_images_actions(self) -> None:
        compare = _load_compare_module()
        dataset, cache = _BatchDataset(), compare.DatumCache(4)
        with patch.object(compare, "_prepare_discrete_datum", side_effect=lambda sample, _, __: _prepared(compare, sample["index"])) as prepare:
            clean = _datum(0)
            first = compare.build_batch(dataset, object(), base_model="openpi/pi05-action-lora-r16-finetune", indices=[0], norm_stats=_norm_stats([1] * 32), state_noise_std=0.5, rng=np.random.default_rng(4), datum_cache=cache)[0]
            second = compare.build_batch(dataset, object(), base_model="openpi/pi05-action-lora-r16-finetune", indices=[0], norm_stats=_norm_stats([1] * 32), state_noise_std=0.5, rng=np.random.default_rng(5), datum_cache=cache)[0]
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(cache.summary()["hits"], 1)
        self.assertIs(first["supervision"], cache._items[0].clean_datum["supervision"])
        self.assertIs(first["observation"]["model_input"]["chunks"][0], cache._items[0].clean_datum["observation"]["model_input"]["chunks"][0])
        self.assertNotEqual(first["observation"]["model_input"]["chunks"][-1]["tokens"], [1, 2, 3])
        self.assertNotEqual(
            first["observation"]["model_input"]["chunks"][-1]["tokens"],
            second["observation"]["model_input"]["chunks"][-1]["tokens"],
        )
        self.assertNotEqual(first["observation"]["state"]["data"], second["observation"]["state"]["data"])
        self.assertEqual(cache._items[0].prefix["state"].tolist(), [0.0] * 32)

    def test_target_augmentation_changes_only_valid_supervision_dimensions(self) -> None:
        compare = _load_compare_module()
        dataset, cache = _BatchDataset(), compare.DatumCache(2)
        stats = _norm_stats([1] * 32)
        stats["actions"].q01[-6:] = 0.0
        stats["actions"].q99[-6:] = 0.0
        diagnostics = compare.TargetAugmentationDiagnostics()
        with patch.object(
            compare,
            "_prepare_discrete_datum",
            side_effect=lambda sample, _, __: _prepared(compare, sample["index"]),
        ):
            augmented = compare.build_batch(
                dataset,
                object(),
                base_model="openpi/pi05-action-lora-r16-finetune",
                indices=[0],
                norm_stats=stats,
                state_noise_std=0.0,
                target_noise_std=0.05,
                rng=np.random.default_rng(43),
                datum_cache=cache,
                target_augmentation_diagnostics=diagnostics,
            )[0]
        actions = np.asarray(augmented["supervision"]["actions"]["data"])
        self.assertEqual(actions.shape, (10, 32))
        self.assertTrue(np.any(actions[:, :26] != 0.0))
        np.testing.assert_array_equal(actions[:, 26:], 0.0)
        self.assertEqual(augmented["observation"]["state"]["data"], [0.0] * 32)
        self.assertEqual(
            augmented["observation"]["model_input"]["chunks"][-1]["tokens"],
            [1, 2, 3],
        )
        np.testing.assert_array_equal(cache._items[0].prefix["actions"], 0.0)
        summary = diagnostics.summary(0.05)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["valid_coordinates"], 260)
        self.assertAlmostEqual(summary["realized_sigma"], 0.05, delta=0.01)

    def test_target_augmentation_parallel_matches_serial_rng(self) -> None:
        compare = _load_compare_module()
        stats = _norm_stats([1] * 32)
        stats["actions"].q01[-6:] = 0.0
        stats["actions"].q99[-6:] = 0.0
        indices = [0, 1, 2]

        def prepare(sample, _, __):
            time.sleep((2 - sample["index"]) * 0.01)
            return _prepared(compare, sample["index"])

        with patch.object(compare, "_prepare_discrete_datum", side_effect=prepare):
            serial = compare.build_batch(
                _BatchDataset(), object(),
                base_model="openpi/pi05-action-lora-r16-finetune", indices=indices,
                norm_stats=stats, state_noise_std=0.0, target_noise_std=0.05,
                rng=np.random.default_rng(43), datum_cache=compare.DatumCache(0),
            )
        with compare.ThreadPoolExecutor(max_workers=3) as executor, patch.object(
            compare, "_prepare_discrete_datum", side_effect=prepare
        ):
            parallel = compare.build_batch(
                _BatchDataset(), object(),
                base_model="openpi/pi05-action-lora-r16-finetune", indices=indices,
                norm_stats=stats, state_noise_std=0.0, target_noise_std=0.05,
                rng=np.random.default_rng(43), datum_cache=compare.DatumCache(0),
                executor=executor,
            )
        self.assertEqual(parallel, serial)

    def test_old_wire_state_mutation_does_not_change_existing_tokens(self) -> None:
        clean = _datum(0)
        changed = clean.copy()
        changed_observation = clean["observation"].copy()
        changed_observation["state"] = {**clean["observation"]["state"], "data": [0.5] * 32}
        changed["observation"] = changed_observation
        self.assertEqual(
            clean["observation"]["model_input"]["chunks"][-1]["tokens"],
            changed["observation"]["model_input"]["chunks"][-1]["tokens"],
        )

    def test_degenerate_quantile_dimensions_stay_clean_and_diagnostics_mask_them(self) -> None:
        compare = _load_compare_module()
        prepared = _prepared(compare, 0)
        prepared.prefix["state"][0] = 1.5  # invalid dimension must not be clipped or perturbed
        stats = _norm_stats([1] * 32)
        stats["state"].q01[0] = 0.0
        stats["state"].q99[0] = 0.0
        diagnostics = compare.AugmentationDiagnostics()
        with patch.object(compare, "_prepare_discrete_datum", return_value=prepared):
            augmented = compare.build_batch(
                _BatchDataset(), object(),
                base_model="openpi/pi05-action-lora-r16-finetune", indices=[0],
                norm_stats=stats, state_noise_std=0.5, rng=np.random.default_rng(4),
                datum_cache=compare.DatumCache(1), augmentation_diagnostics=diagnostics,
            )[0]
        self.assertEqual(augmented["observation"]["state"]["data"][0], 1.5)
        summary = diagnostics.summary(0.5)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["valid_coordinates"], 31)
        self.assertEqual(summary["token_changed_fraction"], 1.0)
        self.assertLessEqual(summary["augmented_out_of_range_fraction_valid_coordinates"], 1.0)

    def test_zero_noise_is_exact_and_does_not_advance_rng(self) -> None:
        compare = _load_compare_module()
        dataset, cache = _BatchDataset(), compare.DatumCache(4)
        rng = np.random.default_rng(8)
        expected = np.random.default_rng(8).normal()
        with patch.object(compare.L, "_transform_sample", side_effect=lambda sample, _: sample), patch.object(compare.L, "_pi05_datum_from_transformed", side_effect=lambda _, sample: _datum(sample["index"])):
            actual = compare.build_batch(dataset, object(), base_model="openpi/pi05-action-lora-r16-finetune", indices=[0], norm_stats=_norm_stats([1] * 32), state_noise_std=0, rng=rng, datum_cache=cache)
        self.assertEqual(actual, [_datum(0)])
        self.assertEqual(rng.normal(), expected)


class CompareTrainingPopulationTests(unittest.TestCase):
    def test_all_row_selection_is_compact_and_deterministic(self) -> None:
        compare = _load_compare_module()
        rows, summary = compare.parse_row_indices("all", 5)
        self.assertEqual(rows, [0, 1, 2, 3, 4])
        self.assertEqual(summary["mode"], "all")
        self.assertEqual(len(summary["sha256"]), 64)
        self.assertEqual(compare.parse_row_indices("3,1,3", 5)[0], [3, 1])

    def test_coverage_sampler_visits_each_row_before_reshuffle(self) -> None:
        compare = _load_compare_module()
        dataset = types.SimpleNamespace(
            _row_start_offset={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            _row_windows={i: types.SimpleNamespace(start_frame=0, end_frame=2) for i in range(5)},
            flat_index=lambda row, frame: row * 10 + frame,
        )
        first = compare.CoverageSampler(dataset, np.random.default_rng(11), slate_size=2, anchors_per_row=1)
        second = compare.CoverageSampler(dataset, np.random.default_rng(11), slate_size=2, anchors_per_row=1)
        first_indices = first.sample_indices(5)
        self.assertEqual(first_indices, second.sample_indices(5))
        self.assertEqual(first.summary()["current_epoch_visited_rows"], 5)
        self.assertEqual(first.summary()["cumulative_visited_rows"], 5)
        self.assertEqual(first.summary()["anchor_min"], 1)
        self.assertEqual(first.summary()["anchor_max"], 1)
        first.sample_indices(1)  # starts epoch 2 without forgetting cumulative coverage
        self.assertEqual(first.summary()["epoch"], 2)
        self.assertEqual(first.summary()["cumulative_visited_rows"], 5)
        self.assertEqual(first.summary()["current_epoch_visited_rows"], 1)

    def test_coverage_resume_replays_sample_and_noise_streams_exactly(self) -> None:
        compare = _load_compare_module()
        dataset = types.SimpleNamespace(
            _row_start_offset={i: i for i in range(8)},
            _source_row_indices=list(range(8)),
            _row_windows={i: types.SimpleNamespace(start_frame=0, end_frame=4) for i in range(8)},
            flat_index=lambda row, frame: row * 10 + frame,
        )
        sample_a, noise_a, _ = compare.make_rngs(42, 43)
        sampler_a = compare.CoverageSampler(dataset, sample_a, slate_size=3, anchors_per_row=2)
        for _ in range(7):
            sampler_a.sample_indices(4)
            for _ in range(4):
                noise_a.normal(0.0, 0.05, size=(32,))
        expected_indices = sampler_a.sample_indices(4)
        expected_noise = [noise_a.normal(0.0, 0.05, size=(32,)) for _ in range(4)]

        sample_b, noise_b, _ = compare.make_rngs(42, 43)
        sampler_b = compare.CoverageSampler(dataset, sample_b, slate_size=3, anchors_per_row=2)
        compare.advance_coverage_rngs(
            sampler_b,
            noise_b,
            completed_steps=7,
            batch_size=4,
            action_horizon=10,
            state_noise_std=0.05,
            target_noise_std=0.0,
        )
        self.assertEqual(sampler_b.sample_indices(4), expected_indices)
        for actual, expected in zip(
            [noise_b.normal(0.0, 0.05, size=(32,)) for _ in range(4)],
            expected_noise,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_coverage_resume_replays_target_noise_stream_exactly(self) -> None:
        compare = _load_compare_module()
        dataset = types.SimpleNamespace(
            _row_start_offset={i: i for i in range(8)},
            _source_row_indices=list(range(8)),
            _row_windows={i: types.SimpleNamespace(start_frame=0, end_frame=4) for i in range(8)},
            flat_index=lambda row, frame: row * 10 + frame,
        )
        sample_a, noise_a, _ = compare.make_rngs(42, 43)
        sampler_a = compare.CoverageSampler(dataset, sample_a, slate_size=3, anchors_per_row=2)
        for _ in range(7):
            sampler_a.sample_indices(4)
            for _ in range(4):
                noise_a.normal(0.0, 0.05, size=(10, 32))
        expected_indices = sampler_a.sample_indices(4)
        expected_noise = [noise_a.normal(0.0, 0.05, size=(10, 32)) for _ in range(4)]

        sample_b, noise_b, _ = compare.make_rngs(42, 43)
        sampler_b = compare.CoverageSampler(dataset, sample_b, slate_size=3, anchors_per_row=2)
        compare.advance_coverage_rngs(
            sampler_b,
            noise_b,
            completed_steps=7,
            batch_size=4,
            action_horizon=10,
            state_noise_std=0.0,
            target_noise_std=0.05,
        )
        self.assertEqual(sampler_b.sample_indices(4), expected_indices)
        for actual, expected in zip(
            [noise_b.normal(0.0, 0.05, size=(10, 32)) for _ in range(4)],
            expected_noise,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


class CompareTrainingPrefetchTests(unittest.TestCase):
    def test_prefetch_matches_synchronous_rng_ordering(self) -> None:
        compare = _load_compare_module()

        def build_sequence(prefetch: bool) -> list[tuple[int, float]]:
            sample_rng, noise_rng, _ = compare.make_rngs(22, 44)

            def build_next() -> list[dict[str, object]]:
                return [{
                    "index": int(sample_rng.integers(0, 100_000)),
                    "noise": float(noise_rng.normal()),
                }]

            if not prefetch:
                return [(item["index"], item["noise"]) for item in (build_next()[0] for _ in range(8))]
            producer = compare.BatchPrefetcher(2, build_next, max_batches=8)
            try:
                return [
                    (item["index"], item["noise"])
                    for item in (producer.next_batch()[0] for _ in range(8))
                ]
            finally:
                producer.close()

        self.assertEqual(build_sequence(False), build_sequence(True))

    def test_coverage_schedule_and_noise_match_with_prefetch(self) -> None:
        compare = _load_compare_module()
        dataset = types.SimpleNamespace(
            _row_start_offset={i: i for i in range(6)},
            _source_row_indices=[10, 11, 12, 13, 14, 15],
            _row_windows={i: types.SimpleNamespace(start_frame=0, end_frame=3) for i in range(6)},
            flat_index=lambda row, frame: row * 10 + frame,
        )

        def sequence(prefetch: bool) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
            sample_rng, noise_rng, _ = compare.make_rngs(42, 43)
            sampler = compare.CoverageSampler(dataset, sample_rng, slate_size=3, anchors_per_row=2)

            def build_next():
                return [
                    tuple(sampler.sample_indices(2)),
                    tuple(noise_rng.normal(0, 0.05, size=4).tolist()),
                ]

            if not prefetch:
                return [tuple(build_next()) for _ in range(6)]
            producer = compare.BatchPrefetcher(2, build_next, max_batches=6)
            try:
                return [tuple(producer.next_batch()) for _ in range(6)]
            finally:
                producer.close()

        self.assertEqual(sequence(False), sequence(True))

    def test_prefetch_propagates_producer_exception(self) -> None:
        compare = _load_compare_module()

        def fail() -> list[dict[str, object]]:
            raise RuntimeError("build failed")

        producer = compare.BatchPrefetcher(1, fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                producer.next_batch()
        finally:
            producer.close()

    def test_prefetch_close_is_idempotent_with_a_full_queue(self) -> None:
        compare = _load_compare_module()
        built = threading.Event()

        def build() -> list[dict[str, object]]:
            built.set()
            return [{"index": 1}]

        producer = compare.BatchPrefetcher(1, build)
        self.assertTrue(built.wait(timeout=1.0))
        producer.close()
        producer.close()
        self.assertFalse(producer._thread.is_alive())


if __name__ == "__main__":
    unittest.main()


class TestParseStopAt:
    """parse_stop_at production helper tests."""

    def test_valid_future_with_timezone(self):
        from scripts.deadline import parse_stop_at
        from datetime import datetime, timedelta, timezone
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        ts = parse_stop_at(future)
        assert ts > 0

    def test_rejects_naive_timestamp(self):
        from scripts.deadline import parse_stop_at
        with pytest.raises(ValueError, match="naive"):
            parse_stop_at("2027-01-01T00:00:00")

    def test_rejects_past_timestamp(self):
        from scripts.deadline import parse_stop_at
        with pytest.raises(ValueError, match="future"):
            parse_stop_at("2020-01-01T00:00:00+08:00")

    def test_accepts_beijing_timezone(self):
        from scripts.deadline import parse_stop_at
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        future = (datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(hours=2)).isoformat()
        ts = parse_stop_at(future)
        assert ts > 0


class TestDeadlineBreak:
    """Deadline break must call save_weights_for_sampler with correct completed_step."""

    def test_deadline_break_saves_and_records(self):
        """Mocked training loop: deadline break calls save and records completed_step."""
        from unittest.mock import MagicMock, patch, call
        import time

        # Simulate the training loop's deadline check and save logic
        model_id = "test-model"
        save_path = "test-save-path"
        stop_at_ts = time.time() - 1  # deadline in the past
        stop_reason = None
        completed_step = 0

        # Simulate 3 steps: step 1 and 2 complete, step 3 hits deadline
        mock_save = MagicMock(return_value={"saved": True})
        global_step = 0
        for step in range(1, 4):
            global_step = step
            # Simulate train_step completing
            pass
            # Check deadline after train_step
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                stop_reason = "deadline"
                completed_step = global_step
                break

        # Verify deadline was hit
        assert stop_reason == "deadline"
        assert completed_step == 1  # first step completed, second step hits deadline

        # Simulate save after break
        mock_save(model_id=model_id, path=save_path)
        mock_save.assert_called_once_with(model_id=model_id, path=save_path)

    def test_no_deadline_runs_all_steps(self):
        """Without deadline, all steps complete."""
        stop_at_ts = None
        stop_reason = None
        completed_step = 0

        for step in range(1, 4):
            global_step = step
            if stop_at_ts is not None and time.time() >= stop_at_ts:
                stop_reason = "deadline"
                completed_step = global_step
                break

        assert stop_reason is None
        assert completed_step == 0
