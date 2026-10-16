"""Per-CLI-version marker packs: one frozen bundle of every marker family (proposal 075).

The driver modules each split *mechanism* from *configuration*: the client,
parser, approval handler, and slash-probe extractor ship the machinery, and the
CLI-version-specific glyphs live in :class:`~maestro_sdk.ClientMarkers`,
:class:`~maestro_sdk.ParserMarkers`, :class:`~maestro_sdk.ApprovalMarkers`, and
:class:`~maestro_sdk.SlashProbeMarkers` — each defaulting to "matches nothing"
so an unconfigured driver honestly degrades (EOF-only turns, everything-is-answer
parsing, no modal detection, every probe parse ``None``). A **marker pack** is
the configuration half: all four families frozen together under a pack ``name``
and an inclusive CLI-version range, so "which layout does this CLI paint" is
answered once, by one versioned object, instead of four independently drifting
marker arguments.

Selection is honest, never a best guess. :func:`select_marker_pack` returns the
newest registered pack whose version range covers the probed CLI version, and
``None`` for an uncovered (or unprobeable) version — the caller then runs with
the empty defaults, exactly today's behaviour, rather than parsing a layout the
pack was never tested against. :func:`detect_cli_version` is the probe half:
a bounded, read-only ``claude --version`` subprocess whose every failure mode
(missing binary, non-zero exit, timeout, unparseable output) degrades to
``None`` rather than raising, mirroring the discipline of Maestro's
``claude auth status`` login probe.

Version ranges compare as dotted-numeric tuples, element-wise and inclusive at
both ends. The comparison is plain tuple ordering, so a shorter bound acts as a
*prefix*: ``max_version="2.1"`` excludes ``"2.1.172"`` (``(2, 1, 172) > (2, 1)``).
Pack authors therefore write explicit full bounds for the range they actually
validated against.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from maestro_sdk.approvals import DEFAULT_APPROVAL_MARKERS, ApprovalMarkers
from maestro_sdk.client import DEFAULT_CLIENT_MARKERS, ClientMarkers
from maestro_sdk.parser import DEFAULT_MARKERS, ParserMarkers
from maestro_sdk.slash_probe import DEFAULT_PROBE_MARKERS, SlashProbeMarkers

# First dotted-numeric run in `claude --version` output ("2.1.172 (Claude Code)").
_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")

# Bound on the --version probe, mirroring maestro's `claude auth status` probe:
# a wedged binary degrades to "version unknown", never a blocked startup.
_PROBE_TIMEOUT_SECONDS = 10.0


def _version_tuple(version: str) -> tuple[int, ...] | None:
    """Parse a dotted-numeric version (``"2.1.172"``) into an int tuple.

    Returns ``None`` for anything that is not purely dotted-numeric — the
    degrade signal :func:`select_marker_pack` treats as "no version, no pack".
    """
    if _VERSION_RE.fullmatch(version.strip()) is None:
        return None
    return tuple(int(part) for part in version.strip().split("."))


def _require_version_tuple(version: str) -> tuple[int, ...]:
    """Parse a pack-author-supplied version bound, failing loud on a bad one.

    A malformed *registered* bound is an authoring bug that must surface at
    pack construction, not silently exclude every CLI version at selection.
    """
    parsed = _version_tuple(version)
    if parsed is None:
        raise ValueError(f"version bound must be dotted-numeric, got {version!r}")
    return parsed


@dataclass(frozen=True)
class MarkerPack:
    """All four marker families frozen together for one CLI-version range.

    ``min_version`` / ``max_version`` bound (inclusively) the CLI versions this
    pack's layout was validated against; both must be dotted-numeric and
    ordered, enforced at construction. Each family defaults to its package
    default ("matches nothing"), so a pack may ship only the families it has
    validated and the rest keep the honest empty-marker degradation.
    """

    name: str
    min_version: str
    max_version: str
    client: ClientMarkers = DEFAULT_CLIENT_MARKERS
    parser: ParserMarkers = DEFAULT_MARKERS
    approvals: ApprovalMarkers = DEFAULT_APPROVAL_MARKERS
    probe: SlashProbeMarkers = DEFAULT_PROBE_MARKERS

    def __post_init__(self) -> None:
        low = _require_version_tuple(self.min_version)
        high = _require_version_tuple(self.max_version)
        if low > high:
            raise ValueError(
                f"min_version {self.min_version!r} exceeds max_version {self.max_version!r}"
            )

    def covers(self, version: str) -> bool:
        """Whether *version* falls inside this pack's inclusive range.

        An unparseable *version* is never covered — refusing to apply a pack
        to a layout it cannot even be compared against.
        """
        parsed = _version_tuple(version)
        if parsed is None:
            return False
        return (
            _require_version_tuple(self.min_version)
            <= parsed
            <= _require_version_tuple(self.max_version)
        )


# Registered packs, oldest-first by convention. Step 6 of proposal 075 ships the
# first real pack; an empty registry keeps select_marker_pack honestly pack-less.
MARKER_PACKS: tuple[MarkerPack, ...] = ()


def get_marker_pack(
    name: str, packs: Sequence[MarkerPack] | None = None
) -> MarkerPack | None:
    """Look up a registered pack by exact name; ``None`` when absent.

    The caller decides whether absence is an error: Maestro's config validation
    fails loud on an unknown *operator-named* pack (a typo must not silently
    mean "no markers"), while ``"auto"`` selection never goes through names.
    """
    for pack in MARKER_PACKS if packs is None else packs:
        if pack.name == name:
            return pack
    return None


def select_marker_pack(
    version: str | None, packs: Sequence[MarkerPack] | None = None
) -> MarkerPack | None:
    """Return the newest registered pack covering *version*, else ``None``.

    ``None`` — for a ``None``/unparseable *version* or an uncovered one — is
    the honest "unknown layout" answer: the caller proceeds with the empty
    default markers (today's EOF-only behaviour) rather than a best-guess pack.
    "Newest" among overlapping covering packs is the highest ``min_version``
    (ties broken by ``max_version``): the pack introduced for the most recent
    layout generation wins over a wider legacy range.
    """
    if version is None:
        return None
    covering = [
        pack
        for pack in (MARKER_PACKS if packs is None else packs)
        if pack.covers(version)
    ]
    if not covering:
        return None
    return max(
        covering,
        key=lambda pack: (
            _require_version_tuple(pack.min_version),
            _require_version_tuple(pack.max_version),
        ),
    )


def detect_cli_version(command: Sequence[str] = ("claude",)) -> str | None:
    """Probe ``<command> --version`` and return the dotted version, or ``None``.

    Read-only and bounded (:data:`_PROBE_TIMEOUT_SECONDS`), mirroring Maestro's
    ``claude auth status`` login-probe discipline. Every failure mode — empty
    *command*, missing binary, OS error, timeout, non-zero exit, or output with
    no dotted-numeric run — returns ``None`` so selection degrades to no-pack
    rather than guessing; converting those specific errors into the documented
    "version unknown" verdict is the contract here, not a silenced defect.
    """
    if not command:
        return None
    try:
        result = subprocess.run(
            [*command, "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    match = _VERSION_RE.search(result.stdout)
    if match is None:
        return None
    return match.group(0)
