"""A detached completion owner survives automatic cleanup through result delivery."""

import shlex
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tools import process_registry as pr
from tui_gateway import server


@pytest.fixture
def background_owner(monkeypatch, tmp_path):
    from agent.client_lifecycle import ClientLifecycleMixin
    import run_agent
    import tools.computer_use.tool as computer_use

    registry = pr.ProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", registry)
    monkeypatch.setattr(pr, "CHECKPOINT_PATH", tmp_path / "processes.json")
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda *a, **k: None)
    monkeypatch.setattr(run_agent, "cleanup_browser", lambda *a, **k: None)
    monkeypatch.setattr(computer_use, "release_computer_use_session", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_emit", lambda *a: None)
    monkeypatch.setattr(server, "_session_pending_kind", lambda *a: None)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(server.threading, "Timer", Timer)
    sid = "background-owner"
    session = dict(session_key="conversation", running=False, history_lock=threading.RLock(),
                   transport=server._detached_ws_transport, created_at=1, last_active=1,
                   agent=SimpleNamespace(_process_owner_task_ids={"turn-owner"}))
    monkeypatch.setattr(server, "_sessions", {sid: session})

    def teardown(popped, **kwargs):
        if popped:
            ClientLifecycleMixin._close_task_resources(popped["agent"], "turn-owner")
            popped["_finalized"] = True

    monkeypatch.setattr(server, "_teardown_popped_session", teardown)
    gate = tmp_path / "release"
    code = ("import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
            "\nwhile not p.exists(): time.sleep(.01)\nprint('BACKGROUND_RESULT')")
    process = registry.spawn_local(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} {shlex.quote(str(gate))}",
        cwd=str(tmp_path), task_id="shared-environment", owner_task_id="turn-owner",
        session_key=session["session_key"])
    process.notify_on_complete = True

    def reap():
        server._schedule_ws_orphan_reap(sid)
        timers[-1].callback()

    try:
        yield SimpleNamespace(registry=registry, process=process, session=session, sid=sid,
                              gate=gate, reap=reap)
    finally:
        if not process.exited:
            registry.kill_process(process.id, source="test_cleanup", consume_output=True)
        if process._reader_thread:
            process._reader_thread.join(timeout=10)


@pytest.mark.parametrize("reaper", ["ws", "ttl", "lru", "backend"])
def test_detached_job_keeps_owner_until_completion_turn(background_owner, monkeypatch, reaper):
    from tools.process_registry_notifications import format_process_notification

    env = background_owner

    def check_protected():
        if reaper == "ws":
            env.reap()
            assert server._sessions.get(env.sid) is env.session
        elif reaper == "ttl":
            assert not server._session_is_evictable(env.sid, env.session, time.time())
        elif reaper == "backend":
            from hermes_cli.web_server_idle_proof import idle_proof

            assert idle_proof(input_probe=lambda: 0)["idle"] is False
        else:
            assert not server._session_is_lru_evictable(env.sid, env.session)

    check_protected()
    assert not env.process.exited
    env.gate.touch()
    event = env.registry.completion_queue.get(timeout=15)
    assert event["exit_code"] == 0
    # A UI status poll is not delivery. Even a dequeued completion must retain its owner.
    assert env.registry.poll(env.process.id)["status"] == "exited"
    check_protected()
    turns = []

    def submit(_rid, sid, session, text, **kwargs):
        assert sid == env.sid and session is env.session
        check_protected()
        turns.append(text)
        server._notif_release_turn(session)

    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    server._notif_handle_ready(env.sid, env.session, [event], set(), env.registry,
                              format_process_notification, None)
    assert len(turns) == 1 and "BACKGROUND_RESULT" in turns[0]
    assert server._session_is_lru_evictable(env.sid, env.session)
    if reaper == "backend":
        from hermes_cli.web_server_idle_proof import idle_proof

        assert idle_proof(input_probe=lambda: 0)["idle"] is True
    env.reap()
    assert env.sid not in server._sessions


@pytest.mark.parametrize("case", ["foreign", "no_notify", "consumed", "stopped", "off", "explicit_close"])
def test_background_protection_does_not_disable_cleanup(background_owner, monkeypatch, case):
    env = background_owner
    if case == "foreign":
        env.session["session_key"] = "another-conversation"
        env.session["agent"]._process_owner_task_ids = {"another-owner"}
    elif case == "no_notify":
        env.process.notify_on_complete = False
    elif case == "consumed":
        env.gate.touch()
        assert env.registry.wait(env.process.id, timeout=15)["status"] == "exited"
    elif case == "stopped":
        env.session["_turn_cancel_requested"] = True
    elif case == "off":
        monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {"background_process_notifications": "off"}})

    if case == "explicit_close":
        server._close_session_by_id(env.sid, end_reason="tui_close")
    else:
        assert server._session_is_lru_evictable(env.sid, env.session)
        env.reap()
    assert env.sid not in server._sessions
    if case not in {"foreign", "consumed"}:
        assert env.process.exited and env.process.termination_source == "agent_close"
