# -*- coding: utf-8 -*-
"""Minimal client for OpenAI-compatible chat completion servers (Ollama, OpenAI, LM Studio, vLLM...)."""
import json
import re
import time

from audiblez.config import api_key, provider_settings


class LLMClient:
    def __init__(self, base_url, model, key='', reasoning_effort=None, timeout=900, retries=3):
        import requests
        self.session = requests.Session()
        if key:
            self.session.headers['Authorization'] = f'Bearer {key}'
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.retries = retries

    def chat_json(self, system, user):
        """Sends one request and returns the parsed JSON object from the reply."""
        payload = dict(model=self.model, temperature=0, response_format={'type': 'json_object'},
                       messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
        if self.reasoning_effort:
            payload['reasoning_effort'] = self.reasoning_effort
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(f'{self.base_url}/chat/completions', json=payload, timeout=self.timeout)
                if r.status_code == 400 and 'reasoning_effort' in payload and 'reasoning' in r.text:
                    payload.pop('reasoning_effort')  # server doesn't support it; retry without
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f'HTTP {r.status_code}: {r.text[:300]}')
                return parse_json(r.json()['choices'][0]['message']['content'])
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    print(f'LLM request failed ({e}), retrying in {2 ** attempt}s...')
                    time.sleep(2 ** attempt)
        raise RuntimeError(f'LLM request to {self.base_url} ({self.model}) failed: {last_error}')


def parse_json(content):
    content = re.sub(r'<think>.*?</think>', '', content or '', flags=re.S).strip()
    content = re.sub(r'^```(?:json)?|```$', '', content, flags=re.M).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find('{'), content.rfind('}')
        if start == -1 or end <= start:
            raise
        return json.loads(content[start:end + 1])


def create_llm(cfg, provider=None, model=None):
    provider = provider or cfg['llm'].get('provider')
    if not provider:
        raise ValueError('No LLM configured. Set [llm] provider and model in the config, or pass --llm-provider/--llm-model.')
    settings = provider_settings(cfg, provider)
    if settings['type'] != 'openai':
        raise ValueError(f'Provider "{provider}" is not an OpenAI-compatible server and cannot be used as the LLM')
    model = model or cfg['llm'].get('model') or settings.get('llm_model')
    if not model:
        raise ValueError(f'No LLM model set. Use --llm-model or set [llm] model in the config.')
    return LLMClient(settings['base_url'], model, key=api_key(settings),
                     reasoning_effort=cfg['llm'].get('reasoning_effort', 'none'))
