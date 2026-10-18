"""Task sandbox adoption uses the same durable session and the real policy scope."""
import asyncio
import copy
import hashlib
import json

import pytest

from tests.gateway.test_session_policy_verbs import _mutation, _owner


def policy_value(cwd):
    value = {'version': 1, 'task_id': 'same-parent', 'launch': {
        'cwd': str(cwd), 'toolsets': ['file', 'terminal'], 'reasoning': 'high',
        'pass_session_id': True, 'run_budget': 90}, 'terminal': {
        'TERMINAL_ENV': 'docker', 'TERMINAL_CWD': str(cwd),
        'TERMINAL_DOCKER_NETWORK': 'false', 'TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE': 'true'},
        'file_roots': {'read': [str(cwd)], 'write': [str(cwd)]},
        'system_prompt': 'FROZEN_TASK_ROLE', 'client_env': {'DOCKER_HOST': 'unix:///task/docker.sock'}}
    value['contract_digest'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return value


@pytest.mark.asyncio
async def test_native_adoption_preserves_history_and_binds_actual_tools(tmp_path, monkeypatch):
    from agent.file_safety import get_read_block_error, get_write_denied_error
    from gateway.session_managed_worker import managed_policy
    from gateway.session_policy import policy_for_source, policy_scope, restore_policy
    from gateway.session_worker_construct import _ephemeral_prompt
    from hermes_state_local import local_receipt
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    try:
        live = authority.sessions[ref.session_id]
        old = policy_for_source(authority.runner, live.source)
        db.append_message(ref.session_id, 'user', 'Retain this original request')
        before = db.get_messages(ref.session_id)
        value = policy_value(tmp_path / 'project-a')
        command = _mutation(db, ref, 'bind', 'task_policy', value)
        rejected = await owner.dispatch(command)
        assert rejected['error']['data']['reason'] == 'permission_denied', rejected
        assert local_receipt(db, ref.session_id)['policy']['request_json'] == old.request_json
        owner.native_owner = True
        accepted = await owner.dispatch(command)
        assert accepted['result']['contract_digest'] == value['contract_digest'], accepted
        assert await owner.dispatch(command) == accepted  # exact durable retry, no second generation
        assert db.get_messages(ref.session_id) == before
        bound = policy_for_source(authority.runner, live.source)
        assert managed_policy(authority, ref) == bound
        assert restore_policy(local_receipt(db, ref.session_id)['policy']) == bound
        assert bound.model == old.model and bound.run_budget_seconds == 90
        assert _ephemeral_prompt(authority, bound, live.source.chat_id) == 'FROZEN_TASK_ROLE'
        inside, outside = tmp_path / 'project-a' / 'code.py', tmp_path / 'outside.py'
        inside.write_text('inside')
        outside.write_text('outside')
        (inside.parent / 'escape').symlink_to(outside)
        with policy_scope(bound):
            from tools.terminal_scope import terminal_env
            assert terminal_env('TERMINAL_ENV') == 'docker'
            assert terminal_env('TERMINAL_DOCKER_NETWORK') == 'false'
            assert get_read_block_error(str(inside)) is None
            assert get_write_denied_error(str(inside)) is None
            assert 'read roots' in get_read_block_error(str(outside))
            assert get_write_denied_error(str(outside)) is not None
            assert get_read_block_error(str(inside.parent / 'escape')) is not None
        with policy_scope(old):
            assert get_read_block_error(str(outside)) is None
            assert get_write_denied_error(str(outside)) is None
        # A generic workspace edit must not detach the task from its sandbox policy.
        moved = await owner.dispatch(_mutation(db, ref, 'move', 'workspace', {'cwd': str(tmp_path)}))
        assert moved['error']['data']['reason'] == 'admission_conflict'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_task_policy_refuses_active_and_queued_input_and_other_task(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from gateway.session_task_policy import digest
    from tests.waiting import async_wait_until
    import hermes_state_runtime as rt
    started, release = asyncio.Event(), asyncio.Event()

    async def handle(event):
        started.set()
        await release.wait()
        return 'ok'

    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    owner.native_owner = True
    value = policy_value(tmp_path / 'project-a')
    try:
        turn = await authority.submit(owner.actor, Submission('one', ref, {'text': 'one'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        queued = await authority.submit(owner.actor, Submission('two', ref, {'text': 'two'}, 'queue'))
        rejected = await owner.dispatch(_mutation(db, ref, 'busy', 'task_policy', value))
        assert rejected['error']['data']['reason'] == 'session_busy', rejected
        release.set()
        await async_wait_until(lambda: rt.get_session_admission(db, admission_id=queued.admission_id)['status'] == 'terminal',
                               what='both admissions to settle')
        assert rt.get_session_admission(db, admission_id=turn.admission_id)['status'] == 'terminal'
        accepted = await owner.dispatch(_mutation(db, ref, 'idle', 'task_policy', value))
        assert 'result' in accepted, accepted
        other = {**value, 'task_id': 'successor'}
        other['contract_digest'] = digest(other)
        rejected = await owner.dispatch(_mutation(db, ref, 'other', 'task_policy', other))
        assert rejected['error']['data']['reason'] == 'admission_conflict', rejected
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.parametrize('field,bad', [('contract_digest', 'forged'), ('version', True),
                                     ('client_env', {'OPENAI_API_KEY': 'forbidden'})])
def test_invalid_task_policy_is_rejected(tmp_path, field, bad):
    from gateway.session_task_policy import validate_task_policy
    from hermes_state_runtime import RuntimeStoreError
    value = copy.deepcopy(policy_value(tmp_path))
    value[field] = bad
    with pytest.raises(RuntimeStoreError):
        validate_task_policy(value)
