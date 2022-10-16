import json
import os
from pathlib import Path
from typing import Dict, Optional

lang_path = Path() / 'lang'
lang_path.mkdir(parents=True, exist_ok=True)

default_path = lang_path / 'default.json'

current: Optional[Dict[str, str]] = None
default: Dict[str, str]


def _save_default():
    with open(default_path, 'w', encoding='utf-8') as f:
        json.dump(default, f, indent=4, ensure_ascii=False)


def _load_default():
    global default
    if default_path.exists():
        if default_path.is_file():
            with open(default_path, 'r', encoding='utf-8') as f:
                default = json.load(f)
        else:
            raise Exception('default.json is a directory')
    else:
        default = {}
        _save_default()


def _marge(lang_dict: dict):
    for key in default:
        if key not in lang_dict:
            lang_dict[key] = "UNLOCALIZED: " + default[key]


def init_translation(current_lang: str = None):
    global current
    global default
    if current_lang is not None or current_lang == 'en_us':
        try:
            with open(lang_path / f'{current_lang}.json', 'r', encoding='utf-8') as f:
                current = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: language file \"{current_lang}.json\" not found, use the default language")

    _load_default()


def translate(original: str) -> str:
    """
    get the translation of the key
    """
    if original not in default:
        default[original] = original
        _save_default()
    if current is None:
        return original
    if original in current:
        return current[original]
    return original


def marge_translation():
    """
    add missed translation to json
    """
    _load_default()

    for lang in os.listdir('lang'):
        if lang != 'default.json' and lang.endswith('.json'):
            with open(f'lang/{lang}', 'r', encoding='utf-8') as f:
                lang_dict = json.load(f)
            _marge(lang_dict)
            with open(f'lang/{lang}', 'w', encoding='utf-8') as f:
                json.dump(lang_dict, f, ensure_ascii=False, indent=4)
