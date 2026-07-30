from __future__ import annotations

import numpy as np
import pytest

from scripts.mano_state44_contract import (
    CONTACT_SLICE,
    FLOOR_SUPPORT_INDEX,
    LIFT_HEIGHT_INDEX,
    MULTICONTACT_PERSISTENCE_INDEX,
    RADIAL_RATE_SLICE,
    STATE44_DIM,
    STATE44_RATE_WINDOW_SECONDS,
    SURFACE_DISTANCE_SLICE,
    State44History,
    assemble_state44_sequence,
    causal_fingertip_radial_rates,
    multicontact_persistence,
    validate_state44_timestamps,
)


def test_state44_layout_is_exactly_44_without_opposition_channel() -> None:
    frames = 3
    state = assemble_state44_sequence(
        hand_qpos=np.arange(frames * 26, dtype=np.float32).reshape(frames, 26),
        contacts=np.ones((frames, 5), dtype=np.float32),
        object_lift=np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        signed_surface_distances=np.full((frames, 5), 0.03, dtype=np.float32),
        radial_rates=np.full((frames, 5), -0.04, dtype=np.float32),
        floor_support=np.asarray([1, 0, 0], dtype=np.float32),
        persistence=np.asarray([0.0, 0.005, 0.01], dtype=np.float32),
    )
    assert state.shape == (frames, STATE44_DIM)
    np.testing.assert_array_equal(state[:, CONTACT_SLICE], 1.0)
    np.testing.assert_allclose(state[:, LIFT_HEIGHT_INDEX], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(state[:, SURFACE_DISTANCE_SLICE], 0.03)
    np.testing.assert_allclose(state[:, RADIAL_RATE_SLICE], -0.04)
    np.testing.assert_array_equal(state[:, FLOOR_SUPPORT_INDEX], [1, 0, 0])
    np.testing.assert_allclose(state[:, MULTICONTACT_PERSISTENCE_INDEX], [0, 0.005, 0.01])
    assert MULTICONTACT_PERSISTENCE_INDEX == 43


def test_radial_rate_is_causal_25ms_and_positive_for_closing() -> None:
    distances = np.full((8, 5), 0.10, dtype=np.float32)
    distances[5:] = 0.095
    rates = causal_fingertip_radial_rates(distances)
    np.testing.assert_array_equal(rates[:5], 0.0)
    np.testing.assert_allclose(
        rates[5], 0.005 / STATE44_RATE_WINDOW_SECONDS, rtol=1e-6
    )
    # A future change cannot modify an already computed earlier observation.
    changed_future = distances.copy()
    changed_future[7] = 0.20
    changed_rates = causal_fingertip_radial_rates(changed_future)
    np.testing.assert_array_equal(changed_rates[:7], rates[:7])
    assert np.all(changed_rates[7] < 0)


def test_state44_history_matches_sequence_rate_and_persistence() -> None:
    radial = np.stack(
        [np.linspace(0.10, 0.08, 12, dtype=np.float32) + finger * 0.001 for finger in range(5)],
        axis=1,
    )
    contacts = np.zeros((12, 5), dtype=np.float32)
    contacts[2:9, :2] = 1.0
    expected_rates = causal_fingertip_radial_rates(radial)
    expected_persistence = multicontact_persistence(contacts)
    history = State44History()
    observed_rates = []
    observed_persistence = []
    for radial_frame, contact_frame in zip(radial, contacts, strict=True):
        rate, persistence = history.observe(radial_frame, contact_frame)
        observed_rates.append(rate)
        observed_persistence.append(persistence)
    np.testing.assert_allclose(observed_rates, expected_rates, atol=1e-7)
    np.testing.assert_allclose(observed_persistence, expected_persistence, atol=1e-7)
    assert expected_persistence[2] == 0.0
    assert expected_persistence[8] == pytest.approx(0.03)
    assert expected_persistence[9] == 0.0


def test_timestamp_contract_rejects_metadata_style_10ms_clock() -> None:
    validate_state44_timestamps(np.arange(6) * 0.005, 6)
    with pytest.raises(ValueError, match="exact monotonic 5 ms"):
        validate_state44_timestamps(np.arange(6) * 0.01, 6)


def test_measurement_geoms_are_noncolliding_and_produce_signed_distances() -> None:
    import mujoco

    from scripts.eval import mano_physics_core as physics

    temporary, model, data, renderer, _object_addr, _object_dof, _hand, _hand_dof, _limits = (
        physics.make_scene(
            "cube1", 32, 32, physics=True, create_renderer=False, state44_features=True
        )
    )
    try:
        assert renderer is None
        tip_ids, object_ids, palm_id = physics.resolve_state44_feature_ids(model, "cube1")
        assert len(tip_ids) == 5
        assert np.all(model.geom_contype[list(tip_ids)] == 0)
        assert np.all(model.geom_conaffinity[list(tip_ids)] == 0)
        mujoco.mj_forward(model, data)
        surface, radial, floor = physics.state44_geometry_from_mujoco(
            model,
            data,
            "cube1",
            tip_geom_ids=tip_ids,
            object_geom_ids=object_ids,
            palm_body_id=palm_id,
        )
        assert surface.shape == radial.shape == (5,)
        assert np.isfinite(surface).all() and np.isfinite(radial).all()
        assert floor in (0.0, 1.0)
    finally:
        temporary.cleanup()
