#!/usr/bin/env python

import os
import sys
import json
from rich import print # pylint: disable=redefined-builtin

if __name__ == "__main__":
    sys.argv.pop(0)
    fn = sys.argv[0] if len(sys.argv) > 0 else 'locale_en.json'
    if not os.path.isfile(fn):
        print(f'File not found: {fn}')
        sys.exit(1)
    with open(fn, 'r', encoding="utf-8") as f:
        data = json.load(f)
    keys = []
    t_names = 0
    t_hints = 0
    t_localized = 0
    t_long = 0
    for k in data.keys():
        names = len(data[k])
        t_names += names
        hints = len([k for k in data[k] if k["hint"] != ""])
        t_hints += hints
        localized = len([k for k in data[k] if k["localized"] != ""])
        t_localized += localized
        missing = names - hints
        long = 0
        for v in data[k]:
            if v['label'] in keys:
                print(f'  Duplicate: {k}.{v["label"]}')
            else:
                if len(v['label']) > 63:
                    long += 1
                    print(f'  Long label: {k}.{v["label"]}')
                keys.append(v['label'])
        t_long += long
        print(f'Section: [bold magenta]{k.ljust(20)}[/bold magenta] entries={names} localized={"[bold green]" + str(localized) + "[/bold green]" if localized > 0 else "0"} long={"[bold red]" + str(long) + "[/bold red]" if long > 0 else "0"} hints={hints} missing={"[bold red]" + str(missing) + "[/bold red]" if missing > 0 else "[bold green]0[/bold green]"}')
    print(f'Totals: entries={t_names} localized={localized} long={t_long} hints={t_hints} missing={t_names - t_hints}')
