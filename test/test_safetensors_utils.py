import json
from unittest.mock import patch

import pytest

from modules.safetensors_utils import read_metadata_from_safetensors


class GuardedFile:
    def __init__(self, file):
        self.file = file
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def fileno(self):
        return self.file.fileno()

    def read(self, size):
        self.read_sizes.append(size)
        # Fail before allocating if malformed input reaches a large read.
        if size < 0 or size > 4096:
            pytest.fail(f"Unbounded metadata read: {size}")
        return self.file.read(size)


@pytest.mark.parametrize("declared_size", [0, 1, 2, 1000, 2**40, 2**46, 2**64 - 1])
def test_rejects_invalid_header_lengths(tmp_path, declared_size):
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(declared_size.to_bytes(8, "little") + b'{"__metadata__": {}}')
    guarded = GuardedFile(path.open("rb"))
    with patch("modules.safetensors_utils.open", return_value=guarded), pytest.raises(ValueError):
        read_metadata_from_safetensors(path)
    assert guarded.read_sizes == [8]


def test_rejects_header_above_safetensors_limit(tmp_path):
    path = tmp_path / "oversized.safetensors"
    declared_size = 100_000_001
    with path.open("wb") as file:
        file.write(declared_size.to_bytes(8, "little") + b'{"')
        file.truncate(8 + declared_size)
    guarded = GuardedFile(path.open("rb"))
    with patch("modules.safetensors_utils.open", return_value=guarded), pytest.raises(ValueError):
        read_metadata_from_safetensors(path)
    assert guarded.read_sizes == [8]


@pytest.mark.parametrize("prefix", [b"", b"\x10", b"\x10" + b"\x00" * 6])
def test_rejects_incomplete_length_prefix(tmp_path, prefix):
    path = tmp_path / "short.safetensors"
    path.write_bytes(prefix)
    with pytest.raises(ValueError):
        read_metadata_from_safetensors(path)


def test_reads_metadata_without_tensor_payload(tmp_path):
    path = tmp_path / "valid.safetensors"
    metadata = {"format": "pt", "nested": '{"value": 1}', "invalid_json": "{bad", "empty": ""}
    header = json.dumps({"__metadata__": metadata}).encode() + b"  "
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"tensor payload")
    guarded = GuardedFile(path.open("rb"))
    with patch("modules.safetensors_utils.open", return_value=guarded):
        result = read_metadata_from_safetensors(path)
    assert result == {**metadata, "nested": {"value": 1}}
    assert sum(guarded.read_sizes) == 8 + len(header)


def test_header_without_metadata_returns_empty_dict(tmp_path):
    path = tmp_path / "no-metadata.safetensors"
    header = b'{"tensor": {}}'
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    assert read_metadata_from_safetensors(path) == {}


def test_invalid_json_is_reported_and_returns_empty_dict(tmp_path, capsys):
    path = tmp_path / "invalid-json.safetensors"
    header = b'{"invalid json}'
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    assert read_metadata_from_safetensors(path) == {}
    assert "Error reading metadata from file" in capsys.readouterr().err
