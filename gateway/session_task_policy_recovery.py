"""Explicit native recovery of a terminal-only mismatch in a frozen task receipt.

The dispatcher never supplies replacement execution settings. The embedded, validated
contract is the sole source; ordinary restore stays fail-closed. The normal mutation
writer owns idle/revision/generation fences, the atomic receipt and publication.
"""
import hashlib
import json


from hermes_state_runtime import RuntimeStoreError, _json

FIELDS = {'task_id', 'contract_digest', 'receipt_digest', 'owner_epoch'}


def validate_recovery_payload(payload):
    if (not isinstance(payload, dict) or set(payload) != FIELDS
            or not isinstance(payload['task_id'], str) or not payload['task_id'].strip()
            or type(payload['owner_epoch']) is not int or payload['owner_epoch'] < 0):
        raise RuntimeStoreError('invalid_params', field='task_policy_recover')
    for name in ('contract_digest', 'receipt_digest'):
        value = payload[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeStoreError('invalid_params', field=name)


def authorize_recovery(authority, actor, ref):
    """Authorize the owned receipt without mounting its invalid execution policy."""
    from gateway.session_authorities import require_serving
    from gateway.session_local_recovery import owned_receipt
    require_serving(authority)
    if actor.profile_id != authority.profile_id or ref.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:control' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    receipt = owned_receipt(authority, ref.session_id)[0]
    if receipt['principal_id'] != actor.subject:
        raise RuntimeStoreError('permission_denied')
    live = authority.sessions.get(ref.session_id)
    if live is not None and live.superseded:
        raise RuntimeStoreError('session_superseded')
    return False  # Never claim imported/foreign history or make a replacement session.


async def prepare_recovery(authority, live, payload, prepared):
    if payload.get('owner_epoch') != authority.epoch:
        raise RuntimeStoreError('stale_epoch')
    return dict(prepared, policy=recovery_candidate(prepared['snapshot']['receipt'], payload))


def recovery_candidate(receipt, payload):
    """Validate the complete exact old receipt and the sole permitted policy delta."""
    from gateway.session_policy import restore_policy
    from gateway.session_task_policy import validate_task_policy
    validate_recovery_payload(payload)
    if hashlib.sha256(_json(receipt).encode()).hexdigest() != payload['receipt_digest']:
        raise RuntimeStoreError('revision_conflict', field='receipt_digest')
    old = receipt['policy']
    try:
        task = validate_task_policy(json.loads(old['request_json'])['task_policy'])
        if task['task_id'] != payload['task_id'] or task['contract_digest'] != payload['contract_digest']:
            raise RuntimeStoreError('admission_conflict')
        # Only the historical refresh mismatch is recoverable, never arbitrary corruption.
        if (old['cwd'] != task['launch']['cwd'] or list(old['toolsets']) != task['launch']['toolsets']
                or json.loads(old['terminal_json']) == task['terminal']):
            raise RuntimeStoreError('admission_conflict')
        candidate = dict(old, terminal_json=json.dumps(task['terminal']))
        restore_policy(candidate)
    except RuntimeStoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc
    # Keep every original field byte-for-byte except terminal_json; no credential rebind,
    # config/profile lookup, transcript write, prompt or external-effect replay.
    return candidate
