"""Approval handler — recognise a tool-approval modal and apply a deny-by-default policy.

When the ``claude`` CLI wants to run a tool it pauses on a modal: a prompt line
plus a numbered list of choices. A bulk-write SDK never sees this — it
pre-authorises tools through the API — but the CLI-driven backend does, and must
decide *mid-stream* whether to type an approval or refuse.

The trust posture is **deny-by-default**: a detected modal with no policy (or a
policy that returns ``False``) raises :class:`ApprovalRequiredError`, surfacing
the decision to the operator rather than guessing. "Approve everything" is one
opt-in predicate away (:func:`approve_all`), but it is never the default — the
operator opts in per session. Sophisticated per-tool policies are out of scope
here; this slice ships the seam and the safe default.

Detection is **mechanism + configuration**, the same split :class:`maestro_sdk.Parser`
uses. The modal's layout — which line is the prompt, how options are numbered,
which option grants approval — is a CLI-version-specific layout policy carried in
:class:`ApprovalMarkers`, not baked into this module. The default markers match
nothing, so without configuration there is no modal to detect: the honest
degenerate behaviour rather than a fragile guess at one CLI version's glyphs.

This module is pure: no I/O, no sleep, no subprocess. Detection is a function of
one screen snapshot; resolution is a function of the request and the policy. The
client slice owns sending the returned keystrokes through the pty.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from maestro_sdk.errors import ApprovalRequiredError


@dataclass(frozen=True)
class ApprovalOption:
    """One selectable choice in an approval modal.

    ``key`` is the literal keystroke string that selects this option — exactly
    what the client types into the pty — and ``label`` is the human-readable text
    beside it (e.g. ``ApprovalOption(key="1", label="Yes")``).
    """

    key: str
    label: str


@dataclass(frozen=True)
class ApprovalRequest:
    """A tool-approval modal recognised on screen.

    Passed to the ``auto_approve`` policy so a predicate can decide on the tool
    name, the prompt, or the option labels; also attached to
    :class:`ApprovalRequiredError` so a denied modal is debuggable without
    reattaching to the pty. ``screen`` is the raw snapshot it was read from.
    """

    prompt: str
    options: tuple[ApprovalOption, ...] = ()
    tool: str | None = None
    screen: str = ""


ApprovalPolicy = Callable[["ApprovalRequest"], bool]
"""A policy is a predicate over a request: ``True`` approves, ``False`` denies."""


def approve_all(request: ApprovalRequest) -> bool:
    """Permissive policy: approve every modal.

    Opt-in only — pass this explicitly to :class:`ApprovalHandler` to authorise
    every tool the CLI asks about. It is never the default; the default (no
    policy) denies by raising.
    """
    return True


@dataclass(frozen=True)
class ApprovalMarkers:
    """Layout policy for finding and parsing an approval modal in a snapshot.

    All matching is against each line *stripped* of surrounding whitespace,
    mirroring :class:`maestro_sdk.ParserMarkers`. Every field defaults to "matches
    nothing," so :data:`DEFAULT_APPROVAL_MARKERS` detects no modal at all.

    - ``prompt``: the detection gate. A snapshot showing a line whose stripped
      form *starts with* any of these strings is an approval modal; the first such
      line becomes :attr:`ApprovalRequest.prompt`. With no prompt markers, nothing
      is ever detected as a modal.
    - ``option``: a regular expression matched against each stripped line; every
      match records an :class:`ApprovalOption`. The pattern must expose a ``key``
      group (the keystroke that selects it) and a ``label`` group (enforced by
      :meth:`compiled_option`).
    - ``tool``: an optional regular expression; the first stripped line that
      matches contributes its ``name`` group as :attr:`ApprovalRequest.tool`.
    - ``approve_keys``: option keys that constitute approval. When a policy
      approves, the handler selects the first option whose key is listed here. An
      empty tuple takes the modal's first option as the approval choice.
    """

    prompt: tuple[str, ...] = ()
    option: str | None = None
    tool: str | None = None
    approve_keys: tuple[str, ...] = ()

    def compiled_option(self) -> re.Pattern[str] | None:
        """Compile :attr:`option` once, or ``None`` if unset.

        Raises :class:`ValueError` if the pattern omits the required ``key`` or
        ``label`` group: :meth:`ApprovalHandler.detect` reads both unconditionally,
        so a missing group is a misconfiguration that must fail loudly at
        construction rather than crash mid-stream during a query.
        """
        if self.option is None:
            return None
        pattern = re.compile(self.option)
        missing = {"key", "label"} - set(pattern.groupindex)
        if missing:
            raise ValueError("option pattern must define (?P<key>...) and (?P<label>...) groups")
        return pattern

    def compiled_tool(self) -> re.Pattern[str] | None:
        """Compile :attr:`tool` once, or ``None`` if unset.

        Raises :class:`ValueError` if the pattern omits the required ``name``
        group, for the same reason as :meth:`compiled_option`.
        """
        if self.tool is None:
            return None
        pattern = re.compile(self.tool)
        if "name" not in pattern.groupindex:
            raise ValueError("tool pattern must define a (?P<name>...) group")
        return pattern


DEFAULT_APPROVAL_MARKERS = ApprovalMarkers()
"""Empty markers: no modal is ever detected, so every snapshot resolves to ``None``."""


class ApprovalHandler:
    """Detects approval modals and applies a deny-by-default policy.

    Construct with the layout :class:`ApprovalMarkers` and an optional
    :data:`ApprovalPolicy`. :meth:`detect` is pure recognition; :meth:`resolve`
    adds the policy decision and returns the keystrokes to type — or raises
    :class:`ApprovalRequiredError`. The client slice calls :meth:`resolve` on each
    fresh snapshot and, when it returns a non-empty string, sends those keystrokes
    through the pty; it must wait for the modal to clear before resolving again so
    a repaint of the same modal is not re-sent.
    """

    def __init__(
        self,
        markers: ApprovalMarkers = DEFAULT_APPROVAL_MARKERS,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        self._markers = markers
        self._policy = policy
        self._option_re = markers.compiled_option()
        self._tool_re = markers.compiled_tool()

    def detect(self, snapshot: str) -> ApprovalRequest | None:
        """Return the modal shown in ``snapshot``, or ``None`` if none is present.

        Pure: scans top to bottom for the first prompt line (the detection gate),
        the first tool-name match, and every option line. A snapshot with no prompt
        line is not a modal.
        """
        prompt: str | None = None
        tool: str | None = None
        options: list[ApprovalOption] = []
        for raw_line in snapshot.split("\n"):
            stripped = raw_line.strip()
            if prompt is None and self._starts_with_any(stripped, self._markers.prompt):
                prompt = stripped
            if tool is None and self._tool_re is not None:
                tool_match = self._tool_re.match(stripped)
                if tool_match is not None:
                    tool = tool_match.group("name")
            if self._option_re is not None:
                option_match = self._option_re.match(stripped)
                if option_match is not None:
                    options.append(
                        ApprovalOption(key=option_match.group("key"), label=option_match.group("label").strip())
                    )
        if prompt is None:
            return None
        return ApprovalRequest(prompt=prompt, options=tuple(options), tool=tool, screen=snapshot)

    def resolve(self, snapshot: str) -> str | None:
        """Detect, consult the policy, and return the keystrokes to type.

        ``None`` when ``snapshot`` shows no modal. Otherwise the keystroke string
        for the approval option when the policy approves. Raises
        :class:`ApprovalRequiredError` when no policy is configured, the policy
        denies, or the policy approves but no approval option is on screen.
        """
        request = self.detect(snapshot)
        if request is None:
            return None
        if self._policy is None:
            raise ApprovalRequiredError(self._deny_message(request, "no policy configured"), screen=snapshot)
        if not self._policy(request):
            raise ApprovalRequiredError(self._deny_message(request, "denied by policy"), screen=snapshot)
        option = self._approval_option(request)
        if option is None:
            raise ApprovalRequiredError(
                self._deny_message(request, "policy approved but no approval option was on screen"),
                screen=snapshot,
            )
        return option.key

    def _approval_option(self, request: ApprovalRequest) -> ApprovalOption | None:
        """The option to select when approving, or ``None`` if there is none.

        With no ``approve_keys`` configured, the first option is the approval
        choice; otherwise the first option whose key is listed.
        """
        if not request.options:
            return None
        if not self._markers.approve_keys:
            return request.options[0]
        for option in request.options:
            if option.key in self._markers.approve_keys:
                return option
        return None

    @staticmethod
    def _deny_message(request: ApprovalRequest, reason: str) -> str:
        tool = f" (tool: {request.tool})" if request.tool else ""
        return f"tool approval required: {reason}{tool}"

    @staticmethod
    def _starts_with_any(stripped: str, markers: tuple[str, ...]) -> bool:
        return any(stripped.startswith(marker) for marker in markers)
