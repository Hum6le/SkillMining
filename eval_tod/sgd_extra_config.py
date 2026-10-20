"""Select the local relay or the existing company workflow client."""
import os

from eval_tod.sgd_local_config import configure as local_configure


def configure(args):
    if getattr(args, 'provider', 'shubiaobiao') == 'shubiaobiao':
        return local_configure(args.env_file, args.model)
    from scripts.launch_sgd_server_batch import workflow_settings, method_environment
    settings = workflow_settings(args, args.workflow_pool_method)
    os.environ.update(method_environment(args, args.workflow_pool_method, os.environ))
    os.environ.setdefault('SKILLMINING_LLM_MAX_RETRIES', '0')
    os.environ.setdefault('SKILLMINING_RETRY_EMPTY_LLM', '1')
    # Imports may already hold the llm module. Bind its public entry points to
    # the company implementation before starting any worker or request.
    import llm
    import llm_new as workflow
    cfg = workflow.get_llm_config()
    if cfg.get('provider') != 'workflow':
        raise ValueError('Company config.py LLM_CONFIG must use provider=workflow')
    for name in ['chat', 'usage_scope', 'get_usage_summary', 'reset_usage_summary',
                 'write_usage_summary', 'resolve_config', 'get_llm_config', 'clear_cache']:
        setattr(llm, name, getattr(workflow, name))
    from workflow_request_pool import from_environment
    if from_environment() is None:
        raise RuntimeError('Request-level workflow pool was not configured')
    return {'provider':'workflow', 'model':args.model or cfg.get('model', 'qwen3.6-flash'),
            'base_url':None, 'enable_thinking':'workflow_managed',
            'workflow_pool_method':args.workflow_pool_method, 'workflow_settings':settings}
