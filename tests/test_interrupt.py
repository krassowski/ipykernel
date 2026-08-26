"""Tests for escalating an interrupt that the running code ignores."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import os
import queue
import signal
import sys
import threading
import time
from contextlib import contextmanager

import IPython
import pytest

from ipykernel.interrupt import MESSAGE, ForcedInterrupt, InterruptEscalator

from .utils import TIMEOUT, get_reply, new_kernel

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="escalation needs sys.monitoring"
)

SETTLE = 0.1
WINDOW = 2.0

SOURCE = """
def swallow_forever(state):
    while True:
        try:
            time.sleep(0.005)
        except BaseException:
            state["caught"] += 1


def swallow_until(state, deadline):
    while time.monotonic() < deadline:
        try:
            time.sleep(0.005)
        except BaseException:
            state["caught"] += 1


def cleanup_and_reraise(state):
    try:
        while True:
            time.sleep(0.005)
    except KeyboardInterrupt:
        state["cleaned"] += 1
        raise


def never_catches(state):
    while True:
        time.sleep(0.005)
"""


def _namespace(filename):
    namespace = {"time": time}
    exec(compile(SOURCE, filename, "exec"), namespace)  # noqa: S102
    return namespace


#: the same code twice, once claiming to be a notebook cell and once claiming
#: to be part of IPython, which is how the escalator tells them apart
USER = _namespace("<user cell>")
KERNEL = _namespace(os.path.join(os.path.dirname(IPython.__file__), "pretend.py"))


@contextmanager
def escalator(settle=SETTLE, window=WINDOW):
    """Arm an escalator on the real SIGINT path, as pre_handler_hook does."""
    reports: list[str] = []
    esc = InterruptEscalator(report=reports.append, settle=settle, window=window)

    def handler(signum, frame):
        esc.note_interrupt()
        signal.default_int_handler(signum, frame)

    saved = signal.signal(signal.SIGINT, handler)
    esc.start()
    try:
        yield reports
    finally:
        esc.stop()
        signal.signal(signal.SIGINT, saved)


def _interrupt_in(delay):
    """Ask the main thread to run its SIGINT handler, as SIGINT would."""
    import _thread

    timer = threading.Timer(delay, _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    return timer


def test_second_interrupt_forces_the_cell_to_stop():
    state = {"caught": 0}
    with escalator() as reports:
        _interrupt_in(0.2)
        _interrupt_in(0.6)
        with pytest.raises(ForcedInterrupt) as excinfo:
            USER["swallow_forever"](state)
    assert reports == [MESSAGE.format(window=WINDOW)]
    assert "<user cell>" in str(excinfo.value)
    assert "swallow_forever" in str(excinfo.value)
    assert state["caught"] == 1


def test_one_interrupt_only_warns():
    """A single ignored interrupt is reported, and the code keeps running."""
    state = {"caught": 0}
    with escalator() as reports:
        _interrupt_in(0.2)
        USER["swallow_until"](state, time.monotonic() + 0.8)
    assert reports == [MESSAGE.format(window=WINDOW)]
    assert state["caught"] == 1


def test_catching_to_clean_up_and_re_raising_is_quiet():
    state = {"cleaned": 0}
    with escalator(settle=0.5) as reports:
        _interrupt_in(0.2)
        with pytest.raises(KeyboardInterrupt):
            USER["cleanup_and_reraise"](state)
    assert reports == []
    assert state["cleaned"] == 1


def test_an_honoured_interrupt_is_quiet():
    with escalator() as reports:
        _interrupt_in(0.2)
        with pytest.raises(KeyboardInterrupt):
            USER["never_catches"]({})
    assert reports == []


def test_a_catch_inside_the_kernel_is_quiet():
    """IPython's own ``except`` means the interrupt worked, not that it was lost."""
    state = {"caught": 0}
    with escalator() as reports:
        _interrupt_in(0.2)
        KERNEL["swallow_until"](state, time.monotonic() + 0.6)
    assert reports == []
    assert state["caught"] == 1


def test_escalation_window_closes():
    """Once the window has passed, the next interrupt starts again from scratch."""
    state = {"caught": 0}
    with escalator(window=0.4) as reports:
        _interrupt_in(0.2)
        _interrupt_in(1.2)
        USER["swallow_until"](state, time.monotonic() + 1.6)
    assert state["caught"] == 2
    assert len(reports) == 2


CELL = """
import time
swallowed = 0
while True:
    try:
        time.sleep(0.05)
    except BaseException:
        swallowed += 1
"""


def _wait_for_stderr(kc, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            msg = kc.get_iopub_msg(timeout=0.5)
        except queue.Empty:
            continue
        if msg["msg_type"] == "stream" and msg["content"]["name"] == "stderr":
            return msg["content"]["text"]
    return ""


def test_forced_interrupt_end_to_end():
    with new_kernel() as kc:
        km = kc.parent
        msg_id = kc.execute(CELL)
        time.sleep(1)  # let the loop get going

        km.interrupt_kernel()
        assert "Interrupt again" in _wait_for_stderr(kc)

        km.interrupt_kernel()
        reply = get_reply(kc, msg_id, TIMEOUT)
        assert reply["content"]["status"] == "error"
        assert reply["content"]["ename"] == "ForcedInterrupt"

        # the kernel survives, and only the first interrupt was swallowed
        msg_id = kc.execute("print(swallowed)")
        get_reply(kc, msg_id, TIMEOUT)
