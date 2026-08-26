"""Escalation for interrupts that the running code refuses to honour."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import sys
import threading
import time
import typing as t
from pathlib import Path
from types import CodeType

# ``sys.monitoring`` reserves tool ids 0 (debugger), 1 (coverage), 2 (profiler)
# and 5 (optimizer). Ids 3 and 4 are free for general use.
TOOL_ID = 4
TOOL_NAME = "ipykernel-interrupt"

#: How long after an ignored interrupt a second one counts as "force it"
#: rather than as a fresh interrupt.
WINDOW = 5.0

#: How long to wait before believing an interrupt was really swallowed.
#: Catching KeyboardInterrupt to clean up and then re-raising is a legitimate
#: pattern, and the kernel should stay quiet about it.
SETTLE = 0.5

MESSAGE = (
    "The running code caught the interrupt and kept going. "
    "Interrupt again within {window:.0f} seconds to force it to stop."
)

_available = sys.version_info >= (3, 12)


class ForcedInterrupt(BaseException):
    """Interrupt delivered after an earlier one was caught and ignored.

    Deliberately not a subclass of :exc:`KeyboardInterrupt`, so that the
    ``except KeyboardInterrupt`` which swallowed the first interrupt does not
    swallow this one as well.
    """


def _line_of(code: CodeType, offset: int) -> int:
    """Return the source line for a bytecode offset.

    ``EXCEPTION_HANDLED`` reports the offset of the handler entry, and that
    instruction carries no line of its own, so fall forward to the first line
    at or after it: the ``except`` clause.
    """
    after: tuple[int, int] | None = None
    for start, end, line in code.co_lines():
        if line is None:
            continue
        if start <= offset < end:
            return line
        if start >= offset and (after is None or start < after[0]):
            after = (start, line)
    return after[1] if after is not None else code.co_firstlineno


def _location(code: CodeType, offset: int) -> str:
    return f"{code.co_filename}, line {_line_of(code, offset)}, in {code.co_name}"


def _kernel_dirs() -> tuple[str, ...]:
    """Directories whose code is the kernel's own, not the user's.

    IPython's ``run_code`` and the kernel's message dispatch both catch
    KeyboardInterrupt on purpose. Those catches mean the interrupt worked.
    """
    # compared against co_filename as plain prefixes, so do not resolve
    # symlinks here: that would stop matching the paths the compiler recorded
    dirs = {str(Path(__file__).parent)}
    for name in ("IPython", "asyncio", "anyio"):
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        if path:
            dirs.add(str(Path(path).parent))
    return tuple(dirs)


class InterruptEscalator:
    """Notice that an interrupt was ignored, and force the next one through.

    A ``KeyboardInterrupt`` is an ordinary exception, so a bare ``except`` or
    ``except BaseException`` swallows it and the cell keeps running. The
    ``EXCEPTION_HANDLED`` event of :mod:`sys.monitoring` reports every caught
    exception, which lets the kernel see that happen.

    Raising from that callback delivers the exception at the moment the
    ``except`` block is entered. That point is outside the ``try`` which caught
    the interrupt, so the raise escapes the loop instead of being caught again.

    Needs Python 3.12. On older versions every method here does nothing.
    """

    def __init__(
        self,
        report: t.Callable[[str], t.Any],
        window: float = WINDOW,
        settle: float = SETTLE,
    ) -> None:
        self._report = report
        self._window = window
        self._settle = settle
        self._kernel_dirs = _kernel_dirs()
        self._thread_ident: int | None = None
        self._claimed = False
        self._watching = False
        self._offered_at: float | None = None
        self._force = False
        self._timer: threading.Timer | None = None

    @property
    def available(self) -> bool:
        """Whether escalation can run at all on this interpreter."""
        return _available

    # -- called from the thread that runs user code -----------------------

    def start(self) -> None:
        """Arm for the message handler that is about to run."""
        if not _available:
            return
        self._thread_ident = threading.get_ident()
        self._reset()

    def stop(self) -> None:
        """Disarm once the message handler is done."""
        if not _available:
            return
        self._thread_ident = None
        self._reset()

    def note_interrupt(self) -> None:
        """Record that an interrupt is being delivered.

        Runs inside the SIGINT handler, so it must not raise.
        """
        if not _available or self._thread_ident is None:
            return
        offered = self._offered_at
        if offered is not None and time.monotonic() - offered <= self._window:
            self._force = True
        self._watch(True)

    # -- sys.monitoring ---------------------------------------------------

    def _on_exception_handled(self, code: CodeType, offset: int, exception: BaseException) -> None:
        if threading.get_ident() != self._thread_ident:
            return
        if not isinstance(exception, (KeyboardInterrupt, ForcedInterrupt)):
            return
        if code.co_filename.startswith(self._kernel_dirs):
            # the kernel or IPython caught it, which means it worked
            return
        if self._force:
            self._force = False
            self._offered_at = None
            self._cancel_timer()
            self._watch(False)
            msg = f"interrupt was caught and ignored at {_location(code, offset)}"
            raise ForcedInterrupt(msg)
        if self._offered_at is None and self._timer is None:
            self._start_timer(self._settle, self._offer)

    def _claim(self) -> bool:
        """Take the monitoring tool id, held only while an interrupt is live."""
        mon = sys.monitoring
        if self._claimed:
            return True
        try:
            mon.use_tool_id(TOOL_ID, TOOL_NAME)
        except ValueError:
            # another tool holds the id; escalation is unavailable
            return False
        mon.register_callback(TOOL_ID, mon.events.EXCEPTION_HANDLED, self._on_exception_handled)
        self._claimed = True
        return True

    def _watch(self, on: bool) -> None:
        """Turn the EXCEPTION_HANDLED event on or off."""
        mon = sys.monitoring
        if on:
            if not self._watching and self._claim():
                mon.set_events(TOOL_ID, mon.events.EXCEPTION_HANDLED)
                self._watching = True
        elif self._watching:
            mon.set_events(TOOL_ID, 0)
            self._watching = False

    def _release(self) -> None:
        """Give the tool id back, so another kernel in this process can use it.

        Never called from the monitoring callback: freeing a tool id while one
        of its own callbacks is on the stack is not worth the risk.
        """
        self._watch(False)
        if self._claimed:
            mon = sys.monitoring
            mon.register_callback(TOOL_ID, mon.events.EXCEPTION_HANDLED, None)
            mon.free_tool_id(TOOL_ID)
            self._claimed = False

    # -- timers -----------------------------------------------------------

    def _start_timer(self, delay: float, target: t.Callable[[], None]) -> None:
        timer = threading.Timer(delay, target)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _offer(self) -> None:
        """Tell the user a second interrupt will be forced through."""
        self._timer = None
        if self._thread_ident is None:
            # the handler finished: the interrupt did its job after all
            return
        self._offered_at = time.monotonic()
        self._start_timer(self._window, self._expire)
        try:
            self._report(MESSAGE.format(window=self._window))
        except Exception:  # noqa: S110
            pass

    def _expire(self) -> None:
        """Close the escalation window and stop watching."""
        self._timer = None
        self._offered_at = None
        self._force = False
        self._release()

    def _reset(self) -> None:
        self._cancel_timer()
        self._release()
        self._offered_at = None
        self._force = False
