"""Load the existing local relay configuration without printing secrets."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT/'third_party/agent-workflow-memory/mind2web/.env'


def configure(env_file=DEFAULT_ENV, model=None):
    from dotenv import dotenv_values
    import llm
    values = dotenv_values(env_file)
    key = values.get('SHUBIAOBIAO_API_KEY') or os.getenv('SHUBIAOBIAO_API_KEY')
    base = values.get('MIND2WEB_API_BASE_URL') or 'https://api.shubiaobiao.cn/v1'
    selected = model or values.get('MIND2WEB_MODEL') or 'qwen3.6-flash'
    if not key:
        raise RuntimeError(f'Missing SHUBIAOBIAO_API_KEY in {env_file}')
    if base.rstrip('/') != 'https://api.shubiaobiao.cn/v1':
        raise ValueError('Unexpected local relay URL; refusing an unintended provider')
    # This domestic relay is reachable directly. A Windows system proxy may
    # reset its TLS connection; bypass only this host in these runner processes.
    for name in ('NO_PROXY', 'no_proxy'):
        hosts = [h.strip() for h in os.environ.get(name, '').split(',') if h.strip()]
        if 'api.shubiaobiao.cn' not in hosts:
            hosts.append('api.shubiaobiao.cn')
        os.environ[name] = ','.join(hosts)
    os.environ['OPENAI_API_KEY'] = key
    os.environ['OPENAI_BASE_URL'] = base
    os.environ['SKILLMINING_ENABLE_THINKING'] = values.get('MIND2WEB_ENABLE_THINKING', 'false')
    os.environ.setdefault('SKILLMINING_LLM_TIMEOUT_SECONDS', '180')
    os.environ.setdefault('SKILLMINING_LLM_MAX_RETRIES', '0')
    os.environ.setdefault('SKILLMINING_RETRY_EMPTY_LLM', '1')
    os.environ['SKILLMINING_REUSE_HTTP_CLIENT'] = '1'
    # Explicit arguments take precedence over unrelated legacy AWM/config.py.
    # Keep this mutation confined to these standalone local runner processes.
    from functools import partial
    if not getattr(llm.chat, '_sgd_local_relay', False):
        original = llm.chat
        llm.chat = partial(original, api_key=key, base_url=base)
        llm.chat._sgd_local_relay = True
    return {'provider': 'shubiaobiao', 'base_url': base, 'model': selected,
            'enable_thinking': os.environ['SKILLMINING_ENABLE_THINKING'],
            'env_file': str(Path(env_file).resolve())}
