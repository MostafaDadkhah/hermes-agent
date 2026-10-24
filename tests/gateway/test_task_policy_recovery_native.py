"""Actual native daemon transport recovery with no submitted prompt or model call."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_normal_runtime_boot import control
from tests.gateway.test_session_task_policy import policy_value


@pytest.mark.parametrize('compressed', [False, True])
def test_native_daemon_recovers_frozen_receipt_without_input(tmp_path, compressed):
    from hermes_state import SessionDB
    from hermes_state_local import local_receipt
    from hermes_state_mutation_compress import compress_in_transaction
    from hermes_state_mutation_prepared import commit_refreshed_policy, local_snapshot
    from hermes_state_runtime import _json

    root = Path(__file__).resolve().parents[2]
    home, user, project = tmp_path / 'home', tmp_path / 'user', tmp_path / 'project'
    for path in (home, user, project):
        path.mkdir(mode=0o700)
    config = {'gateway': {'multiplex_profiles': False},
              'model': {'provider': 'custom', 'default': 'no-inference', 'base_url': 'http://127.0.0.1:9/v1'},
              'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only-unused', PYTHONUNBUFFERED='1')
    policy = policy_value(project)

    async def create(desc):
        async with websocket(home, desc) as ws:
            result = await rpc(ws, 'session.create', request_id='retained-native', source='cli',
                               cwd=str(project), toolsets=[], model='no-inference')
            assert 'result' in result, result
            sid = result['result']['session_id']
            before = await rpc(ws, 'session.resume', session_id=sid)
            snap = before['result']
            bound = await rpc(ws, 'session.mutate', session_id=sid, request_id='native-bind',
                              expected_revision=snap['revision'], expected_generation=snap['execution_generation'],
                              operation='task_policy', payload=policy)
            assert 'result' in bound, bound
            return sid

    with daemon(root, home, env, barrier=False) as (_, desc):
        sid = asyncio.run(create(desc))
    # Fixture construction is confined to the stopped disposable daemon's database.
    with SessionDB(db_path=home / 'state.db') as db:
        epoch = desc['authority_epoch']
        db.append_message(sid, 'user', 'retained original request')
        db.append_message(sid, 'assistant', 'retained partial answer')
        if compressed:
            def compress(conn):
                return compress_in_transaction(db, conn, sid, {}, {
                    'snapshot': local_snapshot(db, conn, sid), 'in_place': False,
                    'messages': [{'role': 'user', 'content': 'retained compressed history'}]})
            db._execute_write(compress)
        receipt = local_receipt(db, sid)
        old = receipt['policy']
        changed = dict(old, terminal_json=json.dumps({'TERMINAL_ENV': 'local', 'TERMINAL_CWD': str(project)}))
        commit_refreshed_policy(db, epoch=epoch, session_id=sid, old=old, new=changed,
                                generation=db.get_session(sid)['runtime_generation'])
        receipt = local_receipt(db, sid)
        before = db.get_session(sid)
        histories = {x: db.get_messages(x) for x in receipt.get('lineage', [sid])}
        payload = {'task_id': policy['task_id'], 'contract_digest': policy['contract_digest'],
                   'receipt_digest': hashlib.sha256(_json(receipt).encode()).hexdigest()}

    async def recover(desc):
        async with websocket(home, desc) as ws:
            failed = await rpc(ws, 'session.resume', session_id=sid)
            assert failed['error']['data']['reason'] == 'storage_unavailable', failed
            request = dict(session_id=sid, request_id='native-recovery', operation='task_policy_recover',
                           expected_revision=before['runtime_revision'],
                           expected_generation=before['runtime_generation'],
                           payload=dict(payload, owner_epoch=desc['authority_epoch']))
            result = await rpc(ws, 'session.mutate', **request)
            assert 'result' in result, result
            readback = await rpc(ws, 'session.resume', session_id=sid)
            assert readback['result']['info']['launch_request']['task_policy'] == policy
            assert readback['result']['pending'] == []
            assert await rpc(ws, 'session.mutate', **request) == result
            return readback['result']

    with daemon(root, home, env, barrier=False) as (_, current):
        readback = asyncio.run(recover(current))
        assert readback['session_id'] == sid
        assert control(home, 'identify')['authority_epoch'] == current['authority_epoch']
        with SessionDB(db_path=home / 'state.db') as db:
            after = local_receipt(db, sid)
            assert after == dict(receipt, policy=dict(receipt['policy'], terminal_json=json.dumps(policy['terminal'])))
            assert {x: db.get_messages(x) for x in histories} == histories
            with db._read_ctx() as conn:
                assert conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
                assert conn.execute('SELECT COUNT(*) FROM worker_executions').fetchone()[0] == 0
