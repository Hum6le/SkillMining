"""One workflow HTTP attempt using the company's provided wire protocol.

No workspace-helper fallback, HTTP retries or automatic redirects. The caller
acquires a workflow permit immediately before each invocation of this module.
"""
from __future__ import annotations

import json
from urllib import request
from urllib.error import HTTPError, URLError

DEFAULT_WORKFLOW_URL = 'http://wgw.huolala.work/proxy/api/open/workflow/run'


class WorkflowHTTPError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class WorkflowRateLimitError(WorkflowHTTPError):
    def __init__(self, message):
        super().__init__(429, message)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_http(req, timeout):
    # Each thread gets its own opener. No hidden redirect may bypass admission.
    return request.build_opener(_NoRedirect()).open(req, timeout=timeout)


def run_flow_once(flow_id, tweaks=None, timeout=None, *, workflow_url=None,
                  server_app_id=None, user_id=None):
    payload = {'workflow_id': flow_id, 'enable_heartbeat': False,
               'user_id': user_id or '{{user_id}}'}
    if tweaks:
        payload['parameters'] = tweaks
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = request.Request(workflow_url or DEFAULT_WORKFLOW_URL, data=body,
        headers={'Content-Type': 'application/json', 'Server-App-Id': server_app_id or 'ios-lma-svc'},
        method='POST')
    try:
        with _open_http(req, timeout or 180) as response:
            data = json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise WorkflowHTTPError(exc.code, f'Workflow HTTP error {exc.code}: {detail[:500]}') from exc
    except URLError as exc:
        raise RuntimeError(f'Workflow request failed: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('Workflow response must be a JSON object')
    ret, message = data.get('ret'), str(data.get('msg', ''))
    # Do not mistake a successful message such as "generated" for a rate limit.
    limited = ret == 1004 or (ret not in (None, 0) and
        ('太快' in message or 'rate' in message.lower() or 'too many requests' in message.lower()))
    if limited:
        raise WorkflowRateLimitError(f'Workflow returned ret={ret}: {message}')
    if ret not in (None, 0):
        raise RuntimeError(f'Workflow returned ret={ret}: {message}')
    return data
