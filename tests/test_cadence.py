"""Tests for the cadence policy.

The plan calls for the cadence layer to be the *only* place timing constants
live, and for tests to assert distribution shape rather than exact wall-clock
behaviour. Every test below seeds the RNG so failures are reproducible.
"""

from __future__ import annotations

import math
import statistics

import pytest

from maestro_sdk import Cadence, InstantCadence
from maestro_sdk._logging import get_logger

# ---------------------------------------------------------------------------
# Helpers


def _sample(cadence: Cadence, prev: str | None, n: int) -> list[float]:
    return [cadence.delay_for("a", prev) for _ in range(n)]


# ---------------------------------------------------------------------------
# delay_for


def test_min_key_floor_is_respected() -> None:
    # mean below the floor with huge stddev → almost every Gaussian draw goes
    # under the floor. The clamp must hold for every sample.
    cadence = Cadence(
        mean_key_ms=5.0,
        stddev_key_ms=500.0,
        min_key_ms=25.0,
        thinking_pause_prob=0.0,
        word_pause_ms=0.0,
        sentence_pause_ms=0.0,
        seed=1,
    )
    floor_s = cadence.min_key_ms / 1000.0
    for delay in _sample(cadence, prev="a", n=2000):
        assert delay >= floor_s


def test_mean_matches_mean_key_ms_over_many_samples() -> None:
    cadence = Cadence(thinking_pause_prob=0.0, seed=42)
    samples = _sample(cadence, prev="a", n=10_000)
    observed_mean_s = statistics.fmean(samples)
    expected_mean_s = cadence.mean_key_ms / 1000.0
    relative_error = abs(observed_mean_s - expected_mean_s) / expected_mean_s
    assert relative_error < 0.05


def test_word_pause_added_after_space() -> None:
    cadence = Cadence(thinking_pause_prob=0.0, seed=2024)
    after_letter = statistics.fmean(_sample(cadence, prev="a", n=10_000))
    after_space = statistics.fmean(_sample(cadence, prev=" ", n=10_000))
    bump_s = after_space - after_letter
    expected_s = cadence.word_pause_ms / 1000.0
    assert math.isclose(bump_s, expected_s, rel_tol=0.05)


def test_word_pause_added_after_tab() -> None:
    cadence = Cadence(thinking_pause_prob=0.0, seed=7)
    after_letter = statistics.fmean(_sample(cadence, prev="a", n=10_000))
    after_tab = statistics.fmean(_sample(cadence, prev="\t", n=10_000))
    bump_s = after_tab - after_letter
    expected_s = cadence.word_pause_ms / 1000.0
    assert math.isclose(bump_s, expected_s, rel_tol=0.05)


@pytest.mark.parametrize("terminator", [".", "!", "?"])
def test_sentence_pause_added_after_terminator(terminator: str) -> None:
    cadence = Cadence(thinking_pause_prob=0.0, seed=99)
    after_letter = statistics.fmean(_sample(cadence, prev="a", n=10_000))
    after_term = statistics.fmean(_sample(cadence, prev=terminator, n=10_000))
    bump_s = after_term - after_letter
    expected_s = cadence.sentence_pause_ms / 1000.0
    assert math.isclose(bump_s, expected_s, rel_tol=0.05)


def test_first_keystroke_has_no_boundary_bonus() -> None:
    cadence = Cadence(thinking_pause_prob=0.0, seed=11)
    samples_first = _sample(cadence, prev=None, n=10_000)
    samples_letter = _sample(cadence, prev="a", n=10_000)
    # prev=None must behave like a non-boundary previous char (no extra pause).
    diff = abs(statistics.fmean(samples_first) - statistics.fmean(samples_letter))
    # Two independent 10k Gaussian means differ by O(stddev/sqrt(n)) ≈ 0.4ms.
    # 5 ms is generous head-room and would not pass if a boundary bump were added.
    assert diff < 0.005


def test_thinking_pause_probability_is_honoured() -> None:
    # Construct a Cadence where (a) the base draw is essentially a constant
    # (tiny stddev) and (b) thinking_pause_ms is far above that constant. A
    # sample classified as "above threshold" can only be a sample where the
    # thinking pause fired.
    cadence = Cadence(
        mean_key_ms=10.0,
        stddev_key_ms=0.0001,
        min_key_ms=0.0,
        word_pause_ms=0.0,
        sentence_pause_ms=0.0,
        thinking_pause_prob=0.5,
        thinking_pause_ms=(1000.0, 3000.0),
        seed=2026,
    )
    samples = _sample(cadence, prev="a", n=20_000)
    threshold_s = 0.500  # well above base (0.01 s), well below pause (1.0 s)
    fired = sum(1 for s in samples if s > threshold_s)
    observed_rate = fired / len(samples)
    assert math.isclose(observed_rate, 0.5, abs_tol=0.02)


def test_thinking_pause_off_means_no_long_delays() -> None:
    cadence = Cadence(
        mean_key_ms=90.0,
        stddev_key_ms=40.0,
        thinking_pause_prob=0.0,
        seed=3,
    )
    samples = _sample(cadence, prev="a", n=5_000)
    # 1 s threshold is ~22 stddevs above mean for a 90+/-40 ms Gaussian; effectively impossible.
    assert max(samples) < 1.0


# ---------------------------------------------------------------------------
# heartbeat_for


def test_heartbeat_zero_chars_returns_min() -> None:
    cadence = Cadence(seed=0)
    assert cadence.heartbeat_for(0) == pytest.approx(cadence.heartbeat_min_ms / 1000.0)


def test_heartbeat_linear_region() -> None:
    cadence = Cadence(seed=0)
    # 100 chars * 12 ms/char = 1200 ms, above the 800 ms floor and below 8000 ms cap.
    assert cadence.heartbeat_for(100) == pytest.approx(1.200)


def test_heartbeat_floor_kicks_in_for_short_responses() -> None:
    cadence = Cadence(seed=0)
    # 50 chars * 12 ms/char = 600 ms; below 800 ms floor → 800 ms.
    assert cadence.heartbeat_for(50) == pytest.approx(0.800)


def test_heartbeat_cap_clamps_long_responses() -> None:
    cadence = Cadence(seed=0)
    assert cadence.heartbeat_for(1_000_000) == pytest.approx(cadence.heartbeat_max_ms / 1000.0)


# ---------------------------------------------------------------------------
# winsize


def test_winsize_within_jitter_envelope() -> None:
    cadence = Cadence(cols=100, rows=32, size_jitter=0.10, seed=5)
    for _ in range(500):
        cols, rows = cadence.winsize()
        assert 90 <= cols <= 110  # ±10 %
        assert 28 <= rows <= 35   # ±10 % then int() floor: 32*1.10=35.2 truncates to 35


def test_winsize_zero_jitter_is_deterministic() -> None:
    cadence = Cadence(cols=80, rows=24, size_jitter=0.0, seed=5)
    assert cadence.winsize() == (80, 24)
    assert cadence.winsize() == (80, 24)


def test_winsize_floors_prevent_unusable_terminal() -> None:
    # cols and rows so small that even no-jitter would underflow the 20-by-8 floor.
    cadence = Cadence(cols=4, rows=2, size_jitter=0.5, seed=5)
    for _ in range(50):
        cols, rows = cadence.winsize()
        assert cols >= 20
        assert rows >= 8


# ---------------------------------------------------------------------------
# Determinism


def test_seeded_cadences_produce_identical_streams() -> None:
    a = Cadence(seed=7)
    b = Cadence(seed=7)
    assert _sample(a, prev="a", n=200) == _sample(b, prev="a", n=200)
    assert a.winsize() == b.winsize()


def test_different_seeds_produce_different_streams() -> None:
    a = Cadence(seed=1)
    b = Cadence(seed=2)
    assert _sample(a, prev="a", n=200) != _sample(b, prev="a", n=200)


def test_unseeded_cadence_is_nondeterministic() -> None:
    # Both Cadences default to seed=None, so each gets system entropy. Streams
    # of any reasonable length should disagree.
    a = Cadence()
    b = Cadence()
    assert _sample(a, prev="a", n=100) != _sample(b, prev="a", n=100)


def test_seed_does_not_affect_equality() -> None:
    a = Cadence(seed=1)
    b = Cadence(seed=1)
    # Advance one of the RNGs; equality must remain stable.
    a.delay_for("x", "x")
    assert a == b


def test_seed_difference_breaks_equality() -> None:
    assert Cadence(seed=1) != Cadence(seed=2)


# ---------------------------------------------------------------------------
# InstantCadence


def test_instant_cadence_delay_is_zero() -> None:
    cadence = InstantCadence()
    assert cadence.delay_for("a", None) == 0.0
    assert cadence.delay_for(" ", ".") == 0.0


def test_instant_cadence_heartbeat_is_zero() -> None:
    cadence = InstantCadence()
    assert cadence.heartbeat_for(0) == 0.0
    assert cadence.heartbeat_for(10_000) == 0.0


def test_instant_cadence_winsize_is_exact() -> None:
    cadence = InstantCadence(cols=77, rows=23)
    assert cadence.winsize() == (77, 23)


def test_instant_cadence_is_a_cadence() -> None:
    assert isinstance(InstantCadence(), Cadence)


# ---------------------------------------------------------------------------
# _logging shim (lives next to cadence, included so the package hits 100% cov)


def test_get_logger_namespaces_under_root() -> None:
    log = get_logger("cadence")
    assert log.name == "maestro_sdk.cadence"
