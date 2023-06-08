#!/usr/bin/env python

import sys
import json
from rich import print # pylint: disable=redefined-builtin

if __name__ == "__main__":
    sys.argv.pop(0)
    fn = sys.argv[0] if len(sys.argv) > 0 else 'locale_en.json'
    with open(fn, 'r') as f:
        data = json.load(f)
    keys = []
    t_names = 0
    t_hints = 0
    for k in data.keys():
        print(f'Section: {k}')
        names = len(data[k])
        t_names += names
        print(f'  Names:   {names}')
        hints = len([k for k in data[k] if k["hint"] != ""])
        t_hints += hints
        print(f'  Hints:   {hints}')
        print(f'  Missing: {names - hints}')
        for v in data[k]:
            if v['text'] in keys:
                print(f'  Duplicate: {k}.{v["text"]}')
            else:
                keys.append(v['text'])
    print(f'Total entries: {t_names}')
    print(f'Total hints:   {t_hints}')
    print(f'Total missing: {t_names - t_hints}')
