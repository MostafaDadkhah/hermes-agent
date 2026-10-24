"""Recovery shares the native transaction, cancellation and publication lifecycle."""
import asyncio
import threading

import pytest

from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_policy_verbs import _mutation, _owner
from tests.gateway.test_session_task_policy import policy_value
from tests.gateway.test_task_policy_recovery import _corrupt_terminal, _ordered_state, _payload


async def _ready(tmp_path, monkeypatch):
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    owner.native_owner = True
    value = policy_value(tmp_path / 'project-a')
    result = await owner.dispatch(_mutation(db, ref, 'bind', 'task_policy', value))
    assert 'result' in result, result
    receipt = _corrupt_terminal(db, authority, ref)
    command = _mutation(db, ref, 'recover', 'task_policy_recover', _payload(receipt, authority))
    return store, db, authority, owner, ref, receipt, command


@pytest.mark.asyncio
async def test_recovery_revalidates_candidate_inside_writer_transaction(tmp_path, monkeypatch):
    from gateway.session_task_policy_recovery import recovery_candidate
    from hermes_state_runtime import mutate_runtime_session
    store, db, authority, owner, ref, receipt, command = await _ready(tmp_path, monkeypatch)
    try:
        params = command['params']
        fields = {k: v for k, v in params.items() if k != 'session_id'}
        fields.update(epoch=authority.epoch, principal_id=owner.actor.subject, session_id=ref.session_id)
        prepared = mutate_runtime_session(db, **fields, _prepare_only=True)
        candidate = recovery_candidate(receipt, params['payload'])
        candidate['safe_mode'] = not candidate['safe_mode']
        prepared['policy'] = candidate
        before = _ordered_state(db)
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            mutate_runtime_session(db, **fields, _prepared=prepared)
        assert _ordered_state(db) == before
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_recovery_retry_publishes_already_committed_receipt_without_second_write(tmp_path, monkeypatch):
    from gateway import session_mutations
    from hermes_state_local import local_receipt
    store, db, authority, owner, ref, receipt, command = await _ready(tmp_path, monkeypatch)
    project = session_mutations._project_committed

    def fail_publication(*args):
        raise RuntimeStoreError('storage_unavailable')

    try:
        monkeypatch.setattr(session_mutations, '_project_committed', fail_publication)
        failed = await owner.dispatch(command)
        assert failed['error']['data']['reason'] == 'storage_unavailable'
        committed = _ordered_state(db)
        assert local_receipt(db, ref.session_id)['policy'] != receipt['policy']
        monkeypatch.setattr(session_mutations, '_project_committed', project)
        retry = await owner.dispatch(command)
        assert 'result' in retry, retry
        assert _ordered_state(db) == committed
        assert ref.session_id in authority.sessions
        assert await owner.dispatch(command) == retry
        assert _ordered_state(db) == committed
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_recovery_survives_request_cancellation_and_retains_no_prompt(tmp_path, monkeypatch):
    from gateway import session_mutations
    from hermes_state_local import local_receipt
    from tests.waiting import async_wait_until
    store, db, authority, owner, ref, receipt, command = await _ready(tmp_path, monkeypatch)
    original = session_mutations.mutate_runtime_session
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        if kwargs.get('_prepare_only'):
            return original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        try:
            return original(*args, **kwargs)
        finally:
            completed.set()

    task = None
    try:
        monkeypatch.setattr(session_mutations, 'mutate_runtime_session', blocked)
        before = _ordered_state(db)
        task = asyncio.create_task(owner.dispatch(command))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        release.set()
        assert await asyncio.to_thread(completed.wait, 5)
        await async_wait_until(lambda: ref.session_id in authority.sessions, what='recovery publication')
        assert local_receipt(db, ref.session_id)['policy'] != receipt['policy']
        after = _ordered_state(db)
        for name in ('messages', 'session_admissions', 'worker_executions'):
            assert after[name] == before[name]
        retry = await owner.dispatch(command)
        assert 'result' in retry, retry
        assert _ordered_state(db) == after
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await owner.close()
        store.close_all_db_handles()
