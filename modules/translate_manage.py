import json
from pathlib import Path
from typing import Dict

lang_path = Path() / 'lang'
lang_path.mkdir(parents=True, exist_ok=True)

en_us: Dict[str, str] = {}
current: Dict[str, str] = {}


def init_translation(current_lang: str = 'en_us'):
    if not (lang_path / 'en_us.json').exists():
        raise ValueError('language file en_us.json not exist')

    with open(lang_path / 'en_us.json', 'r', encoding='utf-8') as f:
        lang = json.load(f)
    en_us.update(lang)

    if current_lang == 'en_us':
        current.update(lang)
    else:
        if not (lang_path / f'{current_lang}.json').exists():
            raise ValueError(f'language file {current_lang}.json not exist')
        with open(lang_path / f'{current_lang}.json', 'r', encoding='utf-8') as f:
            lang = json.load(f)
        current.update(lang)


def translate(key: str) -> str:
    if key in current:
        return current[key]
    elif key in en_us:
        return en_us[key]
    else:
        return key
