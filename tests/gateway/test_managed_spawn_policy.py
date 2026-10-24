"""The admitted policy reaches the real child spawn unchanged, including stop cleanup."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef
from tests.waiting import async_wait_until


CHILD = """import json, os, sys
with open(sys.argv[1], 'w') as out:
    json.dump({k: os.environ.get(k) for k in
               ('HERMES_HOME', 'OPENAI_API_KEY', 'DOCKER_HOST', 'DOCKER_CONFIG')}, out)
sys.stdin.read()
"""


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('kind', ['task', 'ordinary', 'none'])
@pytest.mark.asyncio
async def test_execute_managed_forwards_policy_and_reaps_stopped_child(tmp_path, monkeypatch, kind):
    from gateway import session_managed_worker as managed
    from gateway.session_policy import build_policy
    from tests.gateway.test_session_task_policy import policy_value

    home = tmp_path / 'launch'
    own = home / 'profiles' / 'owner'
    own.mkdir(parents=True)
    (home / '.env').write_text('OPENAI_API_KEY=launch-only-sentinel\n')
    (own / '.env').write_text('OPENAI_API_KEY=own-only-sentinel\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('OPENAI_API_KEY', 'launch-only-sentinel')
    monkeypatch.setenv('DOCKER_HOST', 'unix:///ordinary-only.sock')
    policy = None
    task = policy_value(tmp_path)
    task['client_env'] = {'DOCKER_HOST': 'unix:///task-only.sock', 'DOCKER_CONFIG': str(tmp_path / 'docker')}
    from gateway.session_task_policy import digest
    task['contract_digest'] = digest(task)
    if kind != 'none':
        policy = build_policy({'cwd': str(tmp_path), 'model': 'inert'}, {}, profile_terminal=False)
        if kind == 'task':
            policy = replace(policy, request_json=json.dumps({'task_policy': task}))
    observed = tmp_path / 'child-environment.json'
    original_popen, execution_env = subprocess.Popen, managed._execution_env
    children, forwarded = [], []

    def env_for_policy(authority, actual):
        forwarded.append(actual)
        return execution_env(authority, actual)

    def child(args, **kwargs):
        process = original_popen([sys.executable, '-c', CHILD, str(observed)], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(managed, '_execution_env', env_for_policy)
    monkeypatch.setattr(managed.subprocess, 'Popen', child)
    ref = SessionRef(str(own), 'isolated-session')
    actor = Principal('human', str(own), frozenset({'session:control'}), 'test')
    authority = SimpleNamespace(profile_id=str(own), pending_results={}, waiters={},
        adopt_agent=lambda *args: None, authorize=lambda *args: None,
        check_approval_generation=lambda *args: None)
    row = {'admission_id': 'isolated-admission', 'principal_id': 'human',
           'generation': 3, 'payload': {'text': 'INERT_TEST'}}
    turn = asyncio.create_task(managed.execute_managed(authority, ref, row, policy))
    try:
        await async_wait_until(lambda: turn.done() or (observed.exists() and
            ref.session_id in getattr(authority, '_managed_workers', {})), timeout=15, what='real child spawn')
        if turn.done():
            await turn  # Preserve the actual pre-spawn exception on an unfixed base.
        assert managed.interrupt_managed(authority, actor, ref, 3)
        assert await asyncio.wait_for(turn, 15) == ''
        assert len(forwarded) == 1 and forwarded[0] is policy
        child_env = json.loads(observed.read_text())
        assert child_env['HERMES_HOME'] == str(own)
        assert child_env['OPENAI_API_KEY'] == 'own-only-sentinel'
        if kind == 'task':
            assert {key: child_env[key] for key in task['client_env']} == task['client_env']
        else:
            assert child_env['DOCKER_HOST'] == 'unix:///ordinary-only.sock'
        assert authority.pending_results[row['admission_id']]['result']['interrupted'] is True
        assert not authority._managed_spawning and not authority._managed_workers
        assert len(children) == 1 and children[0].poll() is not None
    finally:
        if not turn.done():
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
