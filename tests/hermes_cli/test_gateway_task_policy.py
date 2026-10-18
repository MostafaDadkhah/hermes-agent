"""A task client must stop before creation/submission when the owner cannot bind its policy."""
import argparse
from contextlib import asynccontextmanager
import json

import pytest

from tests.gateway.test_session_task_policy import policy_value


@pytest.mark.asyncio
async def test_old_owner_is_refused_before_creating_or_submitting(tmp_path, monkeypatch):
    from hermes_cli import gateway_chat
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            assert method == 'runtime.describe'
            return {'capabilities': []}

    @asynccontextmanager
    async def connect():
        yield Peer()

    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy_value(tmp_path)))
    monkeypatch.setattr(gateway_chat, 'connect_gateway', connect)
    args = argparse.Namespace(task_policy_file=str(path), quiet=True, query='TASK', in_dir=str(tmp_path))
    with pytest.raises(gateway_chat.GatewayClientError, match='lacks native task policy'):
        await gateway_chat.run_gateway_chat(args)
    assert calls == ['runtime.describe']


@pytest.mark.asyncio
async def test_binding_requires_exact_owner_readback(tmp_path):
    from hermes_cli.gateway_task_policy import bind_task_policy
    from hermes_cli.gateway_client import GatewayClientError
    policy = policy_value(tmp_path)

    class Peer:
        async def rpc(self, method, **params):
            if method == 'session.mutate':
                return {'contract_digest': policy['contract_digest']}
            assert method == 'session.resume'
            return {'info': {'launch_request': {}}}

    with pytest.raises(GatewayClientError, match='readback differs'):
        await bind_task_policy(Peer(), {'session_id': 'same', 'revision': 1, 'execution_generation': 0}, policy)
