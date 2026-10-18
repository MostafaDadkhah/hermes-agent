"""Finite native-dispatch policy binding on the canonical owner."""
import asyncio
import json
import uuid
from pathlib import Path

from hermes_cli.gateway_client import GatewayClientError


def read_task_policy(path):
    from gateway.session_task_policy import MAX_BYTES, validate_task_policy
    from hermes_state_runtime import RuntimeStoreError
    try:
        with Path(path).expanduser().open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('task policy exceeds size limit')
        return validate_task_policy(json.loads(raw))
    except (OSError, ValueError, RuntimeStoreError) as exc:
        raise GatewayClientError('Invalid --task-policy-file: ' + str(exc)) from exc


async def bind_task_policy(client, snapshot, policy):
    frozen = (snapshot.get('info') or {}).get('launch_request') or {}
    if frozen.get('task_policy') == policy:
        return snapshot
    receipt = await client.rpc('session.mutate', session_id=snapshot['session_id'],
        request_id=uuid.uuid4().hex, expected_revision=snapshot['revision'],
        expected_generation=snapshot['execution_generation'], operation='task_policy', payload=policy)
    if receipt.get('contract_digest') != policy['contract_digest']:
        raise GatewayClientError('Owner did not acknowledge the task execution policy')
    current = await client.rpc('session.resume', session_id=snapshot['session_id'])
    bound = ((current.get('info') or {}).get('launch_request') or {}).get('task_policy')
    if bound != policy:
        raise GatewayClientError('Owner task policy readback differs; no prompt submitted')
    return current


async def prepare_task_launch(args, oneshot):
    from hermes_cli.gateway_chat import _caller_cwd, _requested_policy
    if not getattr(args, 'task_policy_file', None):
        return None
    if not oneshot:
        raise GatewayClientError('--task-policy-file requires a finite run (-Q or --oneshot)')
    task = await asyncio.to_thread(read_task_policy, args.task_policy_file)
    requested = _requested_policy(args)
    for field in ('toolsets', 'reasoning', 'pass_session_id', 'run_budget'):
        if field in requested and requested[field] != task['launch'][field]:
            raise GatewayClientError('--task-policy-file conflicts with ' + field)
    if await asyncio.to_thread(_caller_cwd, args) != task['launch']['cwd']:
        raise GatewayClientError('--task-policy-file conflicts with caller cwd / --in')
    return task


def require_task_capability(description, task):
    from gateway.session_task_policy import CAPABILITY
    if task is not None and CAPABILITY not in description.get('capabilities', []):
        raise GatewayClientError('Gateway lacks native task policy support; update the owner before dispatch')


async def finish_task_launch(client, args, snapshot, task):
    from hermes_cli.gateway_chat import check_resume_policy, continue_title, wants_latest
    if task is not None:
        snapshot = await bind_task_policy(client, snapshot, task)
    if task is not None or wants_latest(args) or getattr(args, 'resume', None) or continue_title(args):
        check_resume_policy(args, snapshot)
    return snapshot
