"""Native dispatcher's explicit execution policy, committed through session.mutate."""
import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

CAPABILITY = 'native-task-policy-v1'
FIELDS = {'version', 'task_id', 'contract_digest', 'launch', 'terminal', 'file_roots',
          'system_prompt', 'client_env'}
LAUNCH_FIELDS = {'cwd', 'toolsets', 'reasoning', 'pass_session_id', 'run_budget'}
MAX_BYTES = 512 * 1024


def digest(policy):
    value = {k: v for k, v in policy.items() if k != 'contract_digest'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _absolute(value):
    return isinstance(value, str) and bool(value.strip()) and Path(value).is_absolute()


def validate_task_policy(value):
    """Validate before any session write; no model, profile or credential override is accepted."""
    try:
        if not isinstance(value, dict) or set(value) != FIELDS or type(value['version']) is not int or value['version'] != 1:
            raise ValueError
        if len(json.dumps(value).encode()) > MAX_BYTES:
            raise ValueError
        if not isinstance(value['task_id'], str) or not value['task_id'].strip() or len(value['task_id']) > 256:
            raise ValueError
        _validate_launch(value['launch'])
        _validate_execution(value)
        _validate_roots(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeStoreError('invalid_params', field='task_policy') from exc
    return value


def _validate_launch(launch):
    if not isinstance(launch, dict) or set(launch) != LAUNCH_FIELDS or not _absolute(launch['cwd']):
        raise ValueError
    tools = launch['toolsets']
    if (not isinstance(tools, list) or not tools or len(tools) != len(set(tools))
            or any(t not in {'file', 'terminal', 'context_engine'} for t in tools)):
        raise ValueError
    from hermes_constants import parse_reasoning_effort
    if (not isinstance(launch['reasoning'], str) or parse_reasoning_effort(launch['reasoning']) is None
            or type(launch['pass_session_id']) is not bool
            or type(launch['run_budget']) is not int or launch['run_budget'] <= 0):
        raise ValueError


def _validate_execution(value):
    launch = value['launch']
    tools = launch['toolsets']
    terminal, env = value['terminal'], value['client_env']
    if (not isinstance(terminal, dict) or not terminal
            or any(not isinstance(k, str) or not k.startswith('TERMINAL_') or not isinstance(v, str)
                   for k, v in terminal.items())
            or terminal.get('TERMINAL_CWD') != launch['cwd']
            or not isinstance(env, dict) or set(env) - {'DOCKER_HOST', 'DOCKER_CONFIG'}
            or any(not isinstance(v, str) or not v for v in env.values())):
        raise ValueError
    if 'terminal' in tools and terminal.get('TERMINAL_ENV') != 'docker':
        raise ValueError


def _validate_roots(value):
    roots = value['file_roots']
    if (not isinstance(roots, dict) or set(roots) != {'read', 'write'}
            or any(not isinstance(paths, list) or not paths or not all(_absolute(p) for p in paths)
                   for paths in roots.values())
            or not isinstance(value['system_prompt'], str)
            or value['contract_digest'] != digest(value)):
        raise ValueError


def task_policy(policy):
    return json.loads(policy.request_json).get('task_policy') if policy is not None else None


async def prepare_task_policy(authority, live, payload, prepared):
    from gateway.session_mutation_policy import _rebind
    from gateway.session_policy import pin_reasoning_effort, restore_policy
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    value = validate_task_policy(payload)
    previous = task_policy(old)
    if previous and previous['task_id'] != value['task_id']:
        raise RuntimeStoreError('admission_conflict')
    if old.kanban_json is not None or old.editor_mcp_json is not None:
        raise RuntimeStoreError('admission_conflict')
    launch = value['launch']
    if not await asyncio.to_thread(Path(launch['cwd']).is_dir):
        raise RuntimeStoreError('workspace_unavailable')
    frozen = old.config()
    pin_reasoning_effort(frozen, launch['reasoning'])
    frozen.setdefault('agent', {})['run_budget_seconds'] = launch['run_budget']
    request = json.loads(old.request_json)
    request.update({k: launch[k] for k in ('toolsets', 'reasoning', 'pass_session_id', 'run_budget')})
    request.update(task_policy=value, follow_profile_config=False)
    policy = replace(old, cwd=launch['cwd'], toolsets=tuple(launch['toolsets']),
                     terminal_json=json.dumps(value['terminal']), config_json=json.dumps(frozen),
                     request_json=json.dumps(request))
    from tui_gateway import git_probe
    git = await asyncio.to_thread(lambda: {'branch': git_probe.branch(launch['cwd']),
                                         'root': git_probe.common_repo_root(launch['cwd'])})
    return _rebind(authority, prepared, old, policy, drop_terminal_secrets=True, git=git)
