# -*- coding: utf-8 -*-
"""User configuration: TTS/LLM providers, default voice and voice metadata.

The config lives in ~/.config/audiblez/config.toml (or $AUDIBLEZ_CONFIG, or --config).
Without a config file audiblez behaves as it always did: local Kokoro with the af_sky voice.
"""
import os
import tomllib
from copy import deepcopy
from pathlib import Path

DEFAULTS = {
    'tts': {'provider': 'kokoro', 'voice': 'af_sky', 'speed': 1.0},
    'llm': {'provider': None, 'model': None},
    'characters': {'prime_replies': True},
    'providers': {'kokoro': {'type': 'kokoro'}},
}

CONFIG_TEMPLATE = '''\
# audiblez configuration
# Providers are named connections. "kokoro" runs locally; "openai" is any server that speaks the
# OpenAI API (/v1/audio/speech for TTS, /v1/chat/completions for the LLM), such as Ollama.

[tts]
provider = "ollama"        # a name from [providers.*] below, or "kokoro"
model = "phonon"           # TTS model on that provider (ignored for kokoro)
voice = "Hugh"             # default narrator voice
speed = 1.0

# The LLM used for character detection (--characters) and narrator suggestions.
# Swap between local and cloud models here or per run with --llm-model, e.g. kimi-k3:cloud
[llm]
provider = "ollama"
model = "gemma4:26b-mlx"

[providers.kokoro]
type = "kokoro"

[providers.ollama]
type = "openai"
base_url = "http://MacBook-Pro-2.local:11434/v1"
# api_key_env = "OLLAMA_API_KEY"   # name of an environment variable holding the API key, if any
# max_chars = 1000                 # longest text sent in a single TTS request

# Voice metadata, used to cast characters by gender and the narrator by accent.
# Kokoro voices are tagged automatically from their names (bm_george = british male).
# gender: male | female        accent: american | british | irish | scottish | australian | ...
# age (optional): child | young | adult | elderly
[providers.ollama.voices]
# Hugh = { gender = "male", accent = "british" }
'''


def config_path(explicit=None):
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get('AUDIBLEZ_CONFIG'):
        return Path(os.environ['AUDIBLEZ_CONFIG']).expanduser()
    base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'audiblez' / 'config.toml'


def _merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(explicit=None):
    cfg = deepcopy(DEFAULTS)
    path = config_path(explicit)
    if path.exists():
        with open(path, 'rb') as f:
            _merge(cfg, tomllib.load(f))
        cfg['_path'] = str(path)
    elif explicit:
        raise FileNotFoundError(f'Config file not found: {path}')
    return cfg


def write_config_template(explicit=None):
    path = config_path(explicit)
    if path.exists():
        raise FileExistsError(f'{path} already exists, not overwriting it')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE, encoding='utf-8')
    return path


def provider_settings(cfg, name):
    if name not in cfg['providers']:
        known = ', '.join(cfg['providers'])
        raise ValueError(f'Unknown provider "{name}". Configured providers: {known}')
    settings = dict(cfg['providers'][name])
    settings.setdefault('type', 'openai')
    settings['name'] = name
    return settings


def api_key(settings):
    if settings.get('api_key'):
        return settings['api_key']
    if settings.get('api_key_env'):
        return os.environ.get(settings['api_key_env'], '')
    return ''
