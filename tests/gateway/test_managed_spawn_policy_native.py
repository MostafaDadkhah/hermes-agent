"""Disposable native create -> policy binding -> submit -> real managed worker, loopback model only."""
import asyncio
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.gateway.test_session_task_policy import policy_value
from tests.waiting import async_wait_until


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('task_bound', [False, True], ids=['ordinary', 'task'])
def test_native_submission_spawns_and_settles_under_accepted_policy(tmp_path, task_bound, record_property):
    root = Path(__file__).resolve().parents[2]
    home, user, project = (tmp_path / name for name in ('profile', 'user', 'project'))
    for path in (home, user, project):
        path.mkdir(mode=0o700)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False, 'managed_workers': True},
              'model': {'provider': 'custom', 'default': 'inert-model', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    policy = policy_value(project)
    text = 'INERT_NATIVE_FORWARDING'

    def rows(sql, args=()):
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return db.execute(sql, args).fetchall()

    async def qualify(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='isolated-create', source='cli',
                                cwd=str(project), model='inert-model', provider='custom', base_url=url, toolsets=[])
            assert 'result' in created, created
            sid = created['result']['session_id']
            if task_bound:
                snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
                bound = await rpc(ws, 'session.mutate', session_id=sid, request_id='isolated-bind',
                    expected_revision=snap['revision'], expected_generation=snap['execution_generation'],
                    operation='task_policy', payload=policy)
                assert 'result' in bound, bound
                accepted = (await rpc(ws, 'session.resume', session_id=sid))['result']
                assert accepted['info']['launch_request']['task_policy'] == policy
            admitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='isolated-submit', text=text)
            assert 'result' in admitted, admitted
            await async_wait_until(lambda: rows('SELECT status FROM session_admissions WHERE request_id=?',
                ('isolated-submit',)) == [('terminal',)], timeout=90, what='managed worker terminal settlement')
            async with asyncio.timeout(15):
                while True:
                    snap = await rpc(ws, 'session.resume', session_id=sid)
                    assert 'result' in snap, snap
                    value = snap['result']
                    if not value['pending']:
                        break
            assert 'RECOVERY_ACK_' + text in json.dumps(value['messages']), value
            assert value['pending'] == []
            if task_bound:
                assert value['info']['launch_request']['task_policy'] == policy
            executions = rows('SELECT session_id,status,owner_epoch,generation FROM worker_executions')
            assert len(executions) == 1 and executions[0][0:2] == (sid, 'terminal'), executions
            assert executions[0][2] == desc['authority_epoch']
            assert rows('SELECT outcome FROM session_admissions') == [('completed',)]
            assert not any(message.get('role') == 'tool' for message in value['messages'])
            assert all(call.get('model') == 'inert-model' for call in peer.requests)
            assert peer.requests
            record_property('native_receipt', json.dumps({'session_id': sid,
                'profile_home': str(home), 'daemon_pid': desc['pid'],
                'owner_epoch': desc['authority_epoch'], 'worker_execution': executions[0],
                'admission': rows('SELECT admission_id,status,outcome,generation FROM session_admissions')[0],
                'task_bound': task_bound, 'contract_digest': policy['contract_digest'] if task_bound else None,
                'loopback_model_calls': len(peer.requests), 'tool_calls': 0}))
            if task_bound:
                assert all('FROZEN_TASK_ROLE' in json.dumps(call['messages']) for call in peer.requests)

    try:
        with daemon(root, home, env, barrier=False) as (process, desc):
            asyncio.run(qualify(desc))
            assert process.poll() is None
            assert json.loads((home / 'config.yaml').read_text()) == config
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
