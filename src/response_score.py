"""Shock-Response Connectivity Score (SRCS)."""

from __future__ import annotations

import numpy as np


def standardize_states(states: np.ndarray) -> np.ndarray:
    """
    Standardize each company's state series independently.

    Parameters
    ----------
    states:
        Array of shape (time, companies).

    Returns
    -------
    np.ndarray
        Standardized array with the same shape.
    """
    mean = states.mean(axis=0)
    std = states.std(axis=0)

    return np.divide(
        states - mean,
        std,
        out=np.zeros_like(states, dtype=float),
        where=std != 0,
    )

def event_response_strength(
    z_states: np.ndarray,
    event_time: int,
    responder: int,
    lag_window: int = 5,
    decay: float = 0.5,
) -> float:
    """
    Calculate the response strength of one company
    after a single shock event.
    """

    total = 0.0

    last_time = min(
        event_time + lag_window,
        z_states.shape[0] - 1,
    )

    for lag in range(1, last_time - event_time + 1):
        response = abs(
            z_states[event_time + lag, responder]
        )

        weight = np.exp(
            -decay * (lag - 1)
        )

        total += weight * response

    return float(total)

def average_response(
    z_states: np.ndarray,
    events: list[tuple[int, int, float]],
    origin: int,
    responder: int,
    lag_window: int = 5,
    decay: float = 0.5,
) -> float:
    """
    Average the response of one company across
    all shocks originating from another company.
    """

    origin_events = [
        event
        for event in events
        if event[1] == origin
    ]

    if not origin_events:
        return 0.0

    scores = []

    for event_time, _, _ in origin_events:
        score = event_response_strength(
            z_states=z_states,
            event_time=event_time,
            responder=responder,
            lag_window=lag_window,
            decay=decay,
        )

        scores.append(score)

    return float(np.mean(scores))

def consistency(
    z_states: np.ndarray,
    events: list[tuple[int, int, float]],
    origin: int,
    responder: int,
    lag_window: int = 5,
    decay: float = 0.5,
    response_threshold: float = 1.0,
) -> float:
    """
    Fraction of shocks from `origin` that are followed
    by a meaningful response from `responder`.
    """

    origin_events = [
        event
        for event in events
        if event[1] == origin
    ]

    if not origin_events:
        return 0.0

    meaningful_responses = 0

    for event_time, _, _ in origin_events:
        score = event_response_strength(
            z_states=z_states,
            event_time=event_time,
            responder=responder,
            lag_window=lag_window,
            decay=decay,
        )

        if score >= response_threshold:
            meaningful_responses += 1

    return meaningful_responses / len(origin_events)

def directed_srcs(
    z_states: np.ndarray,
    events: list[tuple[int, int, float]],
    origin: int,
    responder: int,
    lag_window: int = 5,
    decay: float = 0.5,
    response_threshold: float = 1.0,
) -> float:
    """
    Calculate directed SRCS evidence from origin to responder.
    """

    if origin == responder:
        return 0.0

    average = average_response(
        z_states=z_states,
        events=events,
        origin=origin,
        responder=responder,
        lag_window=lag_window,
        decay=decay,
    )

    consistency_score = consistency(
        z_states=z_states,
        events=events,
        origin=origin,
        responder=responder,
        lag_window=lag_window,
        decay=decay,
        response_threshold=response_threshold,
    )

    return average * consistency_score

def build_srcs_matrix(
    states: np.ndarray,
    events: list[tuple[int, int, float]],
    lag_window: int = 5,
    decay: float = 0.5,
    response_threshold: float = 1.0,
) -> np.ndarray:
    """
    Build the undirected SRCS score matrix.

    Q_ij = (Q_i->j + Q_j->i) / 2
    """

    z_states = standardize_states(states)

    companies = states.shape[1]

    directed = np.zeros((companies, companies))

    for origin in range(companies):
        for responder in range(companies):
            if origin == responder:
                continue

            directed[origin, responder] = directed_srcs(
                z_states=z_states,
                events=events,
                origin=origin,
                responder=responder,
                lag_window=lag_window,
                decay=decay,
                response_threshold=response_threshold,
            )

    # Convert directed evidence into the undirected prototype score.
    srcs = (directed + directed.T) / 2.0

    # No self-connectivity.
    np.fill_diagonal(srcs, 0.0)

    return srcs