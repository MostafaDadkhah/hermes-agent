"""A stored compression child resumes its canonical owner without replaying work."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway import run


@pytest.mark.parametrize('source', ['local', 'gui'])
def test_resume_compression_child_keeps_owner_history_and_authorization(tmp_path, monkeypatch, source):
    async def scenario():
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from gateway.session_authority import initialize_session_authority
        from gateway.session_controls import AuthorityConnection
        from gateway.session_local import create_local_session
        from gateway.session_local_recovery import restore_local_session
        from hermes_state_runtime import list_session_admissions

        monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        db = store._db
        runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                                 _cached_agent_for=lambda route: None)
        runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
        authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
        owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        stranger = AuthorityConnection(authority, object(), {'user_id': 'stranger'})
        try:
            ref = create_local_session(authority, owner.actor, {
                'request_id': 'retained', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
            assert db.try_acquire_compression_lock(ref.session_id, 'fixture')
            try:
                db.publish_compression_child(parent_session_id=ref.session_id, child_session_id='compressed',
                    source=source, messages=[{'role': 'user', 'content': 'retained work'}],
                    compression_lock_holder='fixture')
            finally:
                db.release_compression_lock(ref.session_id, 'fixture')
            restore_local_session(authority, ref.session_id)
            authority.sessions.clear()
            request = {'id': 1, 'method': 'session.resume', 'params': {'session_id': 'compressed'}}
            denied = await stranger.dispatch(request)
            assert denied['error']['message'] == 'permission_denied', denied
            resumed = await owner.dispatch(request)
            assert 'result' in resumed, resumed
            snapshot = resumed['result']
            assert snapshot['session_id'] == snapshot['stored_session_id'] == ref.session_id
            assert [row['content'] for row in snapshot['messages']] == ['retained work']
            assert snapshot['pending'] == [] and not snapshot['running']
            assert list_session_admissions(db, session_id=ref.session_id) == []
            assert db.get_session(ref.session_id)['end_reason'] == 'compression'
        finally:
            await stranger.close()
            await owner.close()
            db.close()

    asyncio.run(scenario())
