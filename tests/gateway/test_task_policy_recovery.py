"""Same-session terminal-only recovery uses native mutation receipts, never prompt replay."""
import hashlib
import json


import pytest

import gateway.session_managed_worker as _session_managed_worker
from gateway import run as _gateway_run
from tests.gateway.test_session_policy_verbs import _mutation, _owner
from tests.gateway.test_session_task_policy import policy_value


def _corrupt_terminal(db, authority, ref):

    from hermes_state_local import local_receipt
    from hermes_state_mutation_prepared import commit_refreshed_policy

    old = local_receipt(db, ref.session_id)['policy']
    changed = dict(old, terminal_json=json.dumps({'TERMINAL_ENV': 'local', 'TERMINAL_CWD': old['cwd']}))
    commit_refreshed_policy(db, epoch=authority.epoch, session_id=ref.session_id, old=old, new=changed,
                            generation=db.get_session(ref.session_id)['runtime_generation'])
    authority.sessions.pop(ref.session_id)
    return local_receipt(db, ref.session_id)


def _payload(receipt, authority):
    from hermes_state_runtime import _json
    task = json.loads(receipt['policy']['request_json'])['task_policy']
    return {'task_id': task['task_id'], 'contract_digest': task['contract_digest'],
            'receipt_digest': hashlib.sha256(_json(receipt).encode()).hexdigest(),
            'owner_epoch': authority.epoch}


@pytest.mark.asyncio
@pytest.mark.parametrize('compressed', [False, True])
async def test_native_recovery_repairs_only_terminal_and_retains_same_history(tmp_path, monkeypatch, compressed):
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    owner.native_owner = True
    try:
        value = policy_value(tmp_path / 'project-a')
        bound = await owner.dispatch(_mutation(db, ref, 'bind-recovery', 'task_policy', value))
        assert 'result' in bound, bound
        db.append_message(ref.session_id, 'user', 'preserved original request')
        db.append_message(ref.session_id, 'assistant', 'preserved partial work')
        if compressed:
            from hermes_state_mutation_compress import compress_in_transaction
            from hermes_state_mutation_prepared import local_snapshot

            def compress(conn):
                snapshot = local_snapshot(db, conn, ref.session_id)
                return compress_in_transaction(db, conn, ref.session_id, {}, {
                    'snapshot': snapshot, 'in_place': False,
                    'messages': [{'role': 'user', 'content': 'retained compression summary'}]})

            _, result = db._execute_write(compress)
            from gateway.session_mutations import _project_committed
            _project_committed(authority, ref, 'compress', result)
        receipt = _corrupt_terminal(db, authority, ref)
        lineage = receipt.get('lineage', [ref.session_id])
        history = {sid: db.get_messages(sid) for sid in lineage}
        original_rows = _ordered_state(db)
        before = db.get_session(ref.session_id)
        payload = _payload(receipt, authority)
        command = _mutation(db, ref, 'recover-one', 'task_policy_recover', payload)
        recovered = await owner.dispatch(command)
        assert 'result' in recovered, recovered
        assert recovered['result']['contract_digest'] == value['contract_digest']
        assert recovered['result']['execution_generation'] == before['runtime_generation'] + 1
        after = local_receipt(db, ref.session_id)
        assert {k: v for k, v in after.items() if k != 'policy'} == {
            k: v for k, v in receipt.items() if k != 'policy'}
        expected = dict(receipt['policy'], terminal_json=json.dumps(value['terminal']))
        assert after['policy'] == expected
        assert restore_policy(after['policy']).cwd == value['launch']['cwd']
        assert {sid: db.get_messages(sid) for sid in lineage} == history
        current_rows = _ordered_state(db)
        for table in ('messages', 'session_admissions', 'worker_executions'):
            assert current_rows[table] == original_rows[table]
        for old_row, new_row in zip(original_rows['sessions'], current_rows['sessions'], strict=True):
            assert {k: v for k, v in old_row.items() if k not in {'runtime_revision', 'runtime_generation'}} == {
                k: v for k, v in new_row.items() if k not in {'runtime_revision', 'runtime_generation'}}
        assert await owner.dispatch(command) == recovered
        assert local_receipt(db, ref.session_id) == after
        assert db.get_session(ref.session_id)['runtime_generation'] == before['runtime_generation'] + 1
        resumed = await owner.dispatch({'id': 'readback', 'method': 'session.resume',
                                       'params': {'session_id': ref.session_id}})
        assert 'result' in resumed, resumed
        assert resumed['result']['info']['launch_request']['task_policy'] == value
        assert resumed['result']['pending'] == []
        mismatch = dict(command, params=dict(command['params'], payload=dict(payload, receipt_digest='0' * 64)))
        refused = await owner.dispatch(mismatch)
        assert refused['error']['data']['reason'] == 'admission_conflict', refused
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.parametrize('mcp_only', [False, True])
def test_direct_profile_refresh_preserves_complete_native_task(tmp_path, monkeypatch, mcp_only):
    from dataclasses import replace
    from gateway.session_policy import build_policy
    from gateway.session_policy_refresh import refreshed_policy
    value = policy_value(tmp_path)
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen',
                           'toolsets': ['file', 'terminal']}, {})
    request = json.loads(policy.request_json)
    request.update(task_policy=value, follow_profile_config=False)
    policy = replace(policy, cwd=str(tmp_path), toolsets=tuple(value['launch']['toolsets']),
                     terminal_json=json.dumps(value['terminal']), request_json=json.dumps(request))
    new, secrets, replaced = refreshed_policy(policy, {'agent': {'max_turns': 3},
        'mcp_servers': {'added': {'command': 'not-launched'}}},
        {'TERMINAL_ENV': 'local', 'TERMINAL_CWD': str(tmp_path)}, mcp_only=mcp_only)
    assert new is policy
    assert secrets == {} and replaced == set()


def _invalid_task(request_json):
    request = json.loads(request_json)
    request['task_policy']['contract_digest'] = 'forged'
    return json.dumps(request)


def _ordered_state(db):
    with db._read_ctx() as conn:
        return {name: [dict(row) for row in conn.execute('SELECT * FROM ' + name + ' ORDER BY rowid')]
                for name in ('sessions', 'messages', 'state_meta', 'session_admissions', 'worker_executions')}


@pytest.mark.asyncio
@pytest.mark.parametrize('fence,reason', [
    ('revision', 'revision_conflict'), ('generation', 'stale_generation'),
    ('epoch', 'stale_epoch'), ('digest', 'revision_conflict'),
    ('task', 'admission_conflict'), ('contract', 'admission_conflict'),
    ('principal', 'permission_denied'), ('profile', 'profile_mismatch'),
    ('capability', 'permission_denied'), ('remote', 'permission_denied'),
    ('cwd', 'admission_conflict'), ('toolsets', 'admission_conflict'),
    ('invalid_policy', 'storage_unavailable'), ('invalid_task', 'invalid_params'),
    ('valid_receipt', 'admission_conflict'), ('queued', 'session_busy'),
    ('unknown', 'unknown_execution'),
])
async def test_recovery_refuses_every_stale_or_unrelated_condition(tmp_path, monkeypatch, fence, reason):
    from dataclasses import replace
    from hermes_state_local import local_receipt
    from hermes_state_mutation_prepared import commit_refreshed_policy
    from hermes_state_runtime import admit_session_input
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    owner.native_owner = True
    try:
        value = policy_value(tmp_path / 'project-a')
        bound = await owner.dispatch(_mutation(db, ref, 'bind-negative', 'task_policy', value))
        assert 'result' in bound, bound
        receipt = _corrupt_terminal(db, authority, ref)
        changes = {
            'cwd': {'cwd': str(tmp_path)},
            'toolsets': {'toolsets': ['terminal']},
            'invalid_policy': {'safe_mode': 'not-a-bool'},
            'invalid_task': {'request_json': _invalid_task(receipt['policy']['request_json'])},
            'valid_receipt': {'terminal_json': json.dumps(value['terminal'])},
        }
        if fence in changes:
            old = receipt['policy']
            commit_refreshed_policy(db, epoch=authority.epoch, session_id=ref.session_id,
                old=old, new=dict(old, **changes[fence]),
                generation=db.get_session(ref.session_id)['runtime_generation'])
            receipt = local_receipt(db, ref.session_id)
        payload = _payload(receipt, authority)
        command = _mutation(db, ref, 'refuse', 'task_policy_recover', payload)
        params = command['params']
        params.update({
            'revision': {'expected_revision': params['expected_revision'] - 1},
            'generation': {'expected_generation': params['expected_generation'] - 1},
        }.get(fence, {}))
        payload.update({
            'epoch': {'owner_epoch': authority.epoch - 1},
            'digest': {'receipt_digest': '0' * 64},
            'task': {'task_id': 'another-task'},
            'contract': {'contract_digest': '0' * 64},
        }.get(fence, {}))
        actor_changes = {
            'principal': {'subject': 'another-principal'},
            'profile': {'profile_id': 'another-profile'},
            'capability': {'capabilities': frozenset({'session:read'})},
        }
        owner.actor = replace(owner.actor, **actor_changes.get(fence, {}))
        owner.native_owner = fence != 'remote'
        if fence in {'queued', 'unknown'}:
            admit_session_input(db, epoch=authority.epoch,
                principal_id=owner.actor.subject, session_id=ref.session_id,
                request_id='pending-no-replay', payload={'text': 'never run'}, intent='queue')
            if fence == 'unknown':
                from hermes_state_runtime import begin_runtime_epoch, claim_session_input, recover_session_inputs
                claim_session_input(db, epoch=authority.epoch, session_id=ref.session_id)
                authority.epoch = begin_runtime_epoch(db, instance_id='replacement-owner')
                recover_session_inputs(db, epoch=authority.epoch)
                payload['owner_epoch'] = authority.epoch
                params['expected_generation'] = db.get_session(ref.session_id)['runtime_generation']
        before = _ordered_state(db)
        refused = await owner.dispatch(command)
        assert refused['error']['data']['reason'] == reason, refused
        assert _ordered_state(db) == before
    finally:
        await owner.close()
        store.close_all_db_handles()
