"""Exception hierarchy.

All exceptions carry an optional ``screen`` snapshot — the visible terminal
buffer at the moment of failure — so failures are debuggable without having to
reattach a debugger to the pty.

Every error also carries a ``category`` string that mirrors, *by name*, the
``ErrorCategory`` enum of Maestro's SDK wrapper (thomas-sodon/maestro-agent) —
``"rate_limit"``, ``"stale_session"``, ``"cli_crash"``, ``"stream_stall"``,
``"timeout"``, ``"auth"``, ``"unknown"``. It is a plain string, not an import of
that enum, so this package stays standalone: Maestro's ``cli_paced`` provider
re-packs ``MaestroSDKError.category`` into a real ``ErrorCategory`` without this
package ever depending on Maestro. Any other caller can map the strings onto
its own retry taxonomy the same way. Each class
declares its natural ``default_category``; a caller can override per-instance for
the cases where the same class spans two categories (a startup timeout is
``"timeout"`` while a mid-stream stall is ``"stream_stall"``, both
:class:`CLITimeoutError`).
"""

from __future__ import annotations


class MaestroSDKError(Exception):
    """Base class for every error raised by :mod:`maestro_sdk`.

    ``default_category`` is the resume-taxonomy bucket this class maps to when no
    explicit ``category`` is passed; subclasses override it. It defaults to
    ``"unknown"`` so an un-categorised failure is honestly opaque rather than
    silently mis-bucketed as something retryable.
    """

    default_category: str = "unknown"

    def __init__(self, message: str, *, screen: str | None = None, category: str | None = None) -> None:
        super().__init__(message)
        self.screen = screen
        self.category = category if category is not None else self.default_category


class CLINotFoundError(MaestroSDKError):
    """The configured CLI command could not be spawned (not found on ``PATH``).

    A missing CLI is an operator-facing credential/precondition failure, not a
    transient runtime error, so it maps to ``"auth"``.
    """

    default_category = "auth"


class CLITimeoutError(MaestroSDKError):
    """The CLI produced no output within the startup budget, or the stream went
    silent before the turn completed.

    The default category is ``"stream_stall"`` (the mid-stream case); the client
    passes ``category="timeout"`` explicitly for the startup case, where the CLI
    never produced any output at all.
    """

    default_category = "stream_stall"


class ApprovalRequiredError(MaestroSDKError):
    """A tool-approval modal was shown but the policy returned ``deny``
    (or no policy was configured).

    Intentionally ``"unknown"``: approval-required is a deny-by-default control
    signal surfaced to the operator, not a transient SDK failure, so there is
    nothing for a caller's retry taxonomy to retry on.
    """


class RateLimitError(MaestroSDKError):
    """The CLI surfaced a rate-limit modal."""

    default_category = "rate_limit"


class SessionExpiredError(MaestroSDKError):
    """The CLI reported the session as expired."""

    default_category = "stale_session"


class CLIExitedError(MaestroSDKError):
    """The pty hit EOF unexpectedly (or the child exited non-zero mid-turn)."""

    default_category = "cli_crash"
