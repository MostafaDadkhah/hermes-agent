"""Real CLI -> native owner -> managed worker -> Docker/file tools (loopback model)."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_session_task_policy import policy_value
from tests.waiting import async_wait_until


class TaskPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        messages = body.get('messages', [])
        task = any(m.get('content') == 'RUN_TASK' for m in messages)
        done = any(m.get('role') == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'TASK_DONE' if task else 'WARM_ACK'}
        if task and not done:
            commands = [('terminal', {'command':
                'cat /workspace/source.txt && test "$(ls /sys/class/net)" = lo && '
                'test ! -e ' + shlex.quote(str(self.server.outside)) +
                ' && printf qualified > /workspace/policy-result.txt'}),
                ('read_file', {'path': str(self.server.outside)}),
                ('write_file', {'path': str(self.server.outside.parent / 'escaped.txt'), 'content': 'ESCAPED'})]
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'task-' + str(i), 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(args)}}
                for i, (name, args) in enumerate(commands)]}
        choice = {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}
        payload = {'id': 'task', 'model': body.get('model'), 'choices': [choice],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        kind = 'application/json'
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            for i, call in enumerate(choice['delta'].get('tool_calls', [])):
                call['index'] = i
            raw = ('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.mark.platforms('posix')
def test_finite_cli_adopts_same_session_and_executes_frozen_task_policy(tmp_path):
    image = os.environ.get('HERMES_TEST_DOCKER_IMAGE')
    docker_host = os.environ.get('HERMES_TEST_DOCKER_HOST')
    if not image or not docker_host:
        pytest.skip('qualification requires an existing local Docker image and daemon')
    from gateway.session_task_policy import digest
    root = Path(__file__).resolve().parents[2]
    home, user, checkout = (tmp_path / name for name in ('state', 'user', 'checkout'))
    for path in (home, user, checkout):
        path.mkdir(mode=0o700)
    outside = tmp_path / 'outside.txt'
    outside.write_text('HOST_SECRET_SENTINEL')
    (checkout / 'source.txt').write_text('TASK_SOURCE_SENTINEL')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), TaskPeer)
    peer.requests, peer.outside = [], outside
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False, 'managed_workers': False},
              'model': {'provider': 'custom', 'default': 'task-model', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}},
              'terminal': {'env_type': 'local'}, 'platform_toolsets': {'cli': ['terminal']}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url,
               DOCKER_HOST='unix:///unusable-owner-default.sock', DOCKER_CONFIG=str(user / 'poison'))
    policy = policy_value(checkout)
    policy['terminal'].update(TERMINAL_DOCKER_IMAGE=image,
        TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES='false', TERMINAL_DOCKER_RUN_AS_HOST_USER='true',
        TERMINAL_DOCKER_VOLUMES=json.dumps([str(checkout) + ':/workspace:rw']))
    policy['file_roots'] = {'read': ['/workspace', str(checkout)], 'write': ['/workspace', str(checkout)]}
    policy['client_env'] = {'DOCKER_HOST': docker_host,
                            'DOCKER_CONFIG': os.environ.get('HERMES_TEST_DOCKER_CONFIG', str(user))}
    policy['contract_digest'] = digest(policy)
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps(policy))

    def rows(sql, args=()):
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return db.execute(sql, args).fetchall()

    async def warm(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='warm-create', source='cli',
                                title='existing-task', cwd=str(user), model='task-model', toolsets=['terminal'])
            assert 'result' in created, created
            sid = created['result']['session_id']
            admitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM')
            assert 'result' in admitted, admitted
            await async_wait_until(lambda: rows('SELECT status FROM session_admissions WHERE request_id=?',
                                                ('warm',)) == [('terminal',)], timeout=60, what='warm turn')
            return sid

    async def readback(desc, sid):
        async with websocket(home, desc) as ws:
            reply = await rpc(ws, 'session.resume', session_id=sid)
            assert 'result' in reply, reply
            return reply['result']

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            sid = asyncio.run(warm(desc))
            result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'chat',
                '-c', 'existing-task', '--create-if-missing', '-Q', '--in', str(checkout),
                '--toolsets', 'file,terminal', '--reasoning', 'high', '--pass-session-id',
                '--run-budget', '90', '--task-policy-file', str(policy_path), '-q', 'RUN_TASK'],
                cwd=root, env=env, capture_output=True, text=True, timeout=150, check=False)
            assert result.returncode == 0, result.stdout + result.stderr + (home / 'restart.log').read_text()
            assert 'TASK_DONE' in result.stdout
            snap = asyncio.run(readback(desc, sid))
            assert snap['stored_session_id'] == sid
            assert snap['info']['launch_request']['task_policy'] == policy
            assert 'WARM_ACK' in json.dumps(snap['messages'])
            assert (checkout / 'policy-result.txt').is_file(), json.dumps(snap['messages']) + result.stderr
            assert (checkout / 'policy-result.txt').read_text() == 'qualified'
            assert outside.read_text() == 'HOST_SECRET_SENTINEL'
            assert not (outside.parent / 'escaped.txt').exists()
            assert rows('SELECT COUNT(*) FROM worker_executions WHERE session_id=?', (sid,)) == [(1,)]
            requests = [r for r in peer.requests if any(m.get('content') == 'RUN_TASK' for m in r.get('messages', []))]
            assert requests and all('FROZEN_TASK_ROLE' in json.dumps(r['messages']) for r in requests)
            final = requests[-1]['messages']
            tool_results = [m for m in final if m['role'] == 'tool']
            assert len(tool_results) == 3 and 'read roots' in json.dumps(tool_results)
            assert 'HOST_SECRET_SENTINEL' not in json.dumps(requests)
            assert json.loads((home / 'config.yaml').read_text()) == config and owner.poll() is None
    finally:
        peer.shutdown()
        peer.server_close()
