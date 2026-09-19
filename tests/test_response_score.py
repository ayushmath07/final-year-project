import numpy as np

from src.response_score import (
    average_response,
    build_srcs_matrix,
    consistency,
    directed_srcs,
    event_response_strength,
    standardize_states,
)

def test_standardize_states_preserves_shape():
    states = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])

    result = standardize_states(states)

    assert result.shape == states.shape


def test_standardize_states_centers_each_company():
    states = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])

    result = standardize_states(states)

    assert np.allclose(result.mean(axis=0), [0.0, 0.0])

def test_event_response_strength_uses_delayed_window():
    states = np.zeros((10, 2))

    # Event at t=4.
    # Responder reacts at t=5.
    states[5, 1] = 2.0

    z_states = standardize_states(states)

    score = event_response_strength(
        z_states=z_states,
        event_time=4,
        responder=1,
        lag_window=3,
        decay=0.5,
    )

    assert score > 0

def test_earlier_response_gets_more_weight():
    early = np.zeros((10, 2))
    late = np.zeros((10, 2))

    # Same response magnitude.
    # Early response occurs at t=5.
    early[5, 1] = 2.0

    # Late response occurs at t=7.
    late[7, 1] = 2.0

    early_z = standardize_states(early)
    late_z = standardize_states(late)

    early_score = event_response_strength(
        early_z,
        event_time=4,
        responder=1,
        lag_window=3,
        decay=0.5,
    )

    late_score = event_response_strength(
        late_z,
        event_time=4,
        responder=1,
        lag_window=3,
        decay=0.5,
    )

    assert early_score > late_score

def test_average_response_uses_only_origin_shocks():
    states = np.zeros((20, 3))

    # Shock/event at t=4 from company 0.
    states[5, 1] = 2.0

    # Another event from company 0 at t=10.
    states[11, 1] = 4.0

    # Event from company 2 should be ignored.
    states[13, 1] = 10.0

    z_states = standardize_states(states)

    events = [
        (4, 0, 1.0),
        (10, 0, 1.0),
        (12, 2, 1.0),
    ]

    result = average_response(
        z_states=z_states,
        events=events,
        origin=0,
        responder=1,
        lag_window=2,
        decay=0.5,
    )

    first = event_response_strength(
        z_states,
        event_time=4,
        responder=1,
        lag_window=2,
        decay=0.5,
    )

    second = event_response_strength(
        z_states,
        event_time=10,
        responder=1,
        lag_window=2,
        decay=0.5,
    )

    expected = (first + second) / 2

    assert np.isclose(result, expected)

def test_consistency_is_fraction_of_meaningful_responses():
    states = np.zeros((20, 2))

    # Event 1 from company 0 -> meaningful response.
    states[5, 1] = 5.0

    # Event 2 from company 0 -> no meaningful response.
    # remains zero

    # Event 3 from company 0 -> meaningful response.
    states[13, 1] = 5.0

    z_states = standardize_states(states)

    events = [
        (4, 0, 1.0),
        (8, 0, 1.0),
        (12, 0, 1.0),
    ]

    result = consistency(
        z_states=z_states,
        events=events,
        origin=0,
        responder=1,
        lag_window=2,
        decay=0.5,
        response_threshold=1.0,
    )

    assert np.isclose(result, 2 / 3)


def test_directed_srcs_combines_average_response_and_consistency():
    states = np.zeros((20, 2))

    # Event 1 from company 0 -> meaningful response.
    states[5, 1] = 5.0

    # Event 2 from company 0 -> meaningful response.
    states[9, 1] = 5.0

    events = [
        (4, 0, 1.0),
        (8, 0, 1.0),
    ]

    z_states = standardize_states(states)

    average = average_response(
        z_states=z_states,
        events=events,
        origin=0,
        responder=1,
        lag_window=2,
        decay=0.5,
    )

    consistency_score = consistency(
        z_states=z_states,
        events=events,
        origin=0,
        responder=1,
        lag_window=2,
        decay=0.5,
        response_threshold=1.0,
    )

    result = directed_srcs(
        z_states=z_states,
        events=events,
        origin=0,
        responder=1,
        lag_window=2,
        decay=0.5,
        response_threshold=1.0,
    )

    assert np.isclose(
        result,
        average * consistency_score,
    )

def test_build_srcs_matrix_is_symmetric_and_has_zero_diagonal():
    states = np.zeros((30, 4))

    # Create responses after shocks from different companies.
    states[6, 1] = 4.0
    states[11, 2] = 3.0
    states[16, 3] = 5.0

    events = [
        (5, 0, 1.0),
        (10, 1, 1.0),
        (15, 2, 1.0),
    ]

    scores = build_srcs_matrix(
        states=states,
        events=events,
        lag_window=2,
        decay=0.5,
        response_threshold=1.0,
    )

    assert scores.shape == (4, 4)
    assert np.allclose(scores, scores.T)
    assert np.allclose(np.diag(scores), 0.0)