#!/usr/bin/env python
import os
import sys
import json
from rich import print # pylint: disable=redefined-builtin


def read_metadata(fn):
    res = {}
    with open(fn, mode="rb") as f:
        metadata_len = f.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = f.read(2)
        if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
            print(f"Not a valid safetensors file: {fn}")
        json_data = json_start + f.read(metadata_len-2)
        json_obj = json.loads(json_data)
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception:
                    pass
    print(f"{fn}: {json.dumps(res, indent=4)}")


def main():
    if len(sys.argv) == 0:
        print('metadata:', 'no files specified')
    for fn in sys.argv:
        if os.path.isfile(fn):
            read_metadata(fn)
        elif os.path.isdir(fn):
            for root, _dirs, files in os.walk(fn):
                for file in files:
                    read_metadata(os.path.join(root, file))

if __name__ == '__main__':
    sys.argv.pop(0)
    main()
