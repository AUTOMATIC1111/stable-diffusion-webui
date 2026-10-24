import os

from modules import errors


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        if len(metadata_len) != 8:
            raise ValueError(f"{filename} has an incomplete safetensors header length")
        metadata_len = int.from_bytes(metadata_len, "little")

        # Match safetensors' 100 MB header limit and check the actual file
        # before passing an untrusted length to read().
        if not 2 < metadata_len <= 100_000_000 or metadata_len > os.fstat(file.fileno()).st_size - 8:
            raise ValueError(f"{filename} has an invalid safetensors header length")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

        res = {}

        try:
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                res[k] = v
                if isinstance(v, str) and v[0:1] == '{':
                    try:
                        res[k] = json.loads(v)
                    except Exception:
                        pass
        except Exception:
            errors.report(f"Error reading metadata from file: {filename}", exc_info=True)

        return res
