"""Event-conditioned Shock-Response Connectivity Score (SRCS)."""

from __future__ import annotations

import numpy as np

Event = tuple[int, int, float]


def _validate_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 2:
        raise ValueError("states must have shape (time >= 2, companies >= 2)")
    if not np.isfinite(states).all():
        raise ValueError("states must be finite")
    return states


def standardize_states(states: np.ndarray) -> np.ndarray:
    """Column-standardize observations; constant columns become zeros."""
    states = _validate_states(states)
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    return np.divide(states - mean, std, out=np.zeros_like(states), where=std > 0)


def _valid_events(events: list[Event], time_count: int, companies: int, lag_window: int) -> list[Event]:
    if lag_window < 1:
        raise ValueError("lag_window must be positive")
    valid: list[Event] = []
    for time, company, magnitude in events:
        if not 0 <= int(company) < companies:
            raise ValueError("event company is out of range")
        if not np.isfinite(magnitude):
            raise ValueError("event magnitude must be finite")
        if 0 <= int(time) < time_count - lag_window:
            valid.append((int(time), int(company), float(magnitude)))
    return valid


def event_response_strength(z_states: np.ndarray, event_time: int, responder: int,
                            lag_window: int = 5, decay: float = 0.5,
                            event_sign: float | None = None) -> float:
    """Return the largest lag-decayed signed-consistent response in a full window."""
    z_states = _validate_states(z_states)
    if lag_window < 1 or decay < 0:
        raise ValueError("lag_window must be positive and decay non-negative")
    if not 0 <= responder < z_states.shape[1]:
        raise ValueError("responder is out of range")
    if event_time < 0 or event_time + lag_window >= z_states.shape[0]:
        return float("nan")
    responses = z_states[event_time + 1:event_time + lag_window + 1, responder]
    if event_sign not in (None, 0):
        responses = np.maximum(responses * np.sign(event_sign), 0.0)
    else:
        responses = np.abs(responses)
    weights = np.exp(-decay * np.arange(lag_window, dtype=float))
    return float(np.max(weights * responses))


def event_response_scores(z_states: np.ndarray, events: list[Event], origin: int,
                          responder: int, lag_window: int = 5,
                          decay: float = 0.5) -> np.ndarray:
    valid = _valid_events(events, z_states.shape[0], z_states.shape[1], lag_window)
    return np.asarray([
        event_response_strength(z_states, time, responder, lag_window, decay, magnitude)
        for time, company, magnitude in valid if company == origin
    ], dtype=float)


def average_response(z_states: np.ndarray, events: list[Event], origin: int, responder: int,
                     lag_window: int = 5, decay: float = 0.5) -> float:
    scores = event_response_scores(z_states, events, origin, responder, lag_window, decay)
    return float(scores.mean()) if scores.size else 0.0


def consistency(z_states: np.ndarray, events: list[Event], origin: int, responder: int,
                lag_window: int = 5, decay: float = 0.5,
                response_threshold: float = 1.0) -> float:
    if response_threshold < 0:
        raise ValueError("response_threshold must be non-negative")
    scores = event_response_scores(z_states, events, origin, responder, lag_window, decay)
    return float(np.mean(scores >= response_threshold)) if scores.size else 0.0


def directed_srcs(z_states: np.ndarray, events: list[Event], origin: int, responder: int,
                  lag_window: int = 5, decay: float = 0.5,
                  response_threshold: float = 1.0) -> float:
    if origin == responder:
        return 0.0
    scores = event_response_scores(z_states, events, origin, responder, lag_window, decay)
    if not scores.size:
        return 0.0
    return float(scores.mean() * np.mean(scores >= response_threshold))


def build_srcs_matrix(states: np.ndarray, events: list[Event], lag_window: int = 5,
                      decay: float = 0.5, response_threshold: float = 1.0) -> np.ndarray:
    """Build a finite symmetric SRCS matrix without using ground truth."""
    states = _validate_states(states)
    z_states = standardize_states(states)
    companies = states.shape[1]
    directed = np.zeros((companies, companies), dtype=float)
    for origin in range(companies):
        for responder in range(companies):
            if origin != responder:
                directed[origin, responder] = directed_srcs(
                    z_states, events, origin, responder, lag_window, decay, response_threshold
                )
    srcs = (directed + directed.T) / 2.0
    np.fill_diagonal(srcs, 0.0)
    return srcs


def event_evidence_table(states: np.ndarray, events: list[Event], origin: int, responder: int,
                         lag_window: int = 5, decay: float = 0.5) -> list[dict[str, float | int]]:
    """Return auditable per-event evidence for one directed pair."""
    z_states = standardize_states(states)
    valid = _valid_events(events, len(states), states.shape[1], lag_window)
    rows = []
    for time, company, magnitude in valid:
        if company == origin:
            rows.append({"event_time": time, "origin": origin, "responder": responder,
                         "shock": magnitude, "response_strength": event_response_strength(
                             z_states, time, responder, lag_window, decay, magnitude)})
    return rows
