"""Cadence policy — the heart of :mod:`maestro_sdk`.

Everything that makes this SDK *human-paced* lives here:

* per-key typing delay, drawn from a clamped Gaussian with word/sentence
  pauses and an occasional "thinking pause";
* post-response heartbeat scaled to answer length;
* pty winsize, randomised per session so the CLI cannot detect a fixed shape.

There is no ``sleep`` call anywhere else in the package — every wait in the
package routes through ``delay_for`` or ``heartbeat_for``. Cadence is frozen
once constructed: tests that need different rhythm construct a fresh instance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_WORD_BOUNDARIES = frozenset({" ", "\t"})
_SENTENCE_BOUNDARIES = frozenset({".", "!", "?"})


@dataclass(frozen=True)
class Cadence:
    """Pacing policy.

    All ``_ms`` fields are in milliseconds (so the configuration reads
    naturally for humans); every method returns a delay in **seconds** so it
    can feed straight into :func:`asyncio.sleep` without a unit conversion at
    the call site.
    """

    mean_key_ms: float = 90.0
    stddev_key_ms: float = 40.0
    min_key_ms: float = 25.0
    word_pause_ms: float = 250.0
    sentence_pause_ms: float = 600.0
    thinking_pause_prob: float = 0.03
    thinking_pause_ms: tuple[float, float] = (1000.0, 3000.0)

    heartbeat_min_ms: float = 800.0
    heartbeat_per_answer_char_ms: float = 12.0
    heartbeat_max_ms: float = 8000.0

    cols: int = 100
    rows: int = 32
    size_jitter: float = 0.10

    seed: int | None = None

    # The RNG is internal mutable state: frozen=True prevents callers from
    # *rebinding* a Cadence's rhythm mid-flight (which would invalidate the
    # heartbeat invariant), but it does not — and must not — prevent the RNG
    # from advancing as we draw samples. The repr/eq excludes it so two
    # Cadences with the same seed compare equal regardless of draw history.
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", random.Random(self.seed))

    def delay_for(self, char: str, prev: str | None) -> float:
        """Delay in seconds before pressing ``char``, given ``prev`` was just pressed.

        The base draw is a Gaussian clamped at ``min_key_ms``. If ``prev`` was
        a word boundary, ``word_pause_ms`` is added; if a sentence terminator,
        ``sentence_pause_ms``. With probability ``thinking_pause_prob`` an
        additional uniform pause in ``thinking_pause_ms`` is added — modelling
        the operator pausing to think mid-sentence.
        """
        del char  # reserved for per-character variation in a future slice; do not remove
        delay_ms = max(self.min_key_ms, self._rng.gauss(self.mean_key_ms, self.stddev_key_ms))
        if prev in _WORD_BOUNDARIES:
            delay_ms += self.word_pause_ms
        elif prev in _SENTENCE_BOUNDARIES:
            delay_ms += self.sentence_pause_ms
        if self._rng.random() < self.thinking_pause_prob:
            low, high = self.thinking_pause_ms
            delay_ms += self._rng.uniform(low, high)
        return delay_ms / 1000.0

    def heartbeat_for(self, answer_chars: int) -> float:
        """Delay in seconds after the CLI finishes streaming, before another send is allowed.

        Scales linearly with answer length, floored at ``heartbeat_min_ms``
        and capped at ``heartbeat_max_ms``.
        """
        delay_ms = max(self.heartbeat_min_ms, self.heartbeat_per_answer_char_ms * answer_chars)
        delay_ms = min(delay_ms, self.heartbeat_max_ms)
        return delay_ms / 1000.0

    def winsize(self) -> tuple[int, int]:
        """Return ``(cols, rows)`` for the pty, jittered ±``size_jitter`` per call.

        Small floors (20 cols, 8 rows) prevent jitter from producing a window
        the CLI cannot draw into.
        """
        jitter_cols = 1.0 + self._rng.uniform(-self.size_jitter, self.size_jitter)
        jitter_rows = 1.0 + self._rng.uniform(-self.size_jitter, self.size_jitter)
        cols = max(20, int(self.cols * jitter_cols))
        rows = max(8, int(self.rows * jitter_rows))
        return cols, rows


class InstantCadence(Cadence):
    """All delays zero; winsize deterministic.

    For unit tests only. :class:`AsyncClient` logs a WARNING on each query when
    it sees an :class:`InstantCadence`, so accidentally shipping a no-pacing
    client to production is loud, not silent.
    """

    def delay_for(self, char: str, prev: str | None) -> float:
        del char, prev
        return 0.0

    def heartbeat_for(self, answer_chars: int) -> float:
        del answer_chars
        return 0.0

    def winsize(self) -> tuple[int, int]:
        return self.cols, self.rows
