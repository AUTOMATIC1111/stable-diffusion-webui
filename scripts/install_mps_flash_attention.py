#!/usr/bin/env python3
"""Install Metal Flash SDPA with the PyTorch 2.3 MPS stream safety backport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


PACKAGE = "mps-flash-sdpa"
VERSION = "0.1.0"


def replace_exact(path, old, new, expected_count=1):
    text = path.read_text()
    count = text.count(old)
    if count != expected_count:
        raise RuntimeError(f"Expected {expected_count} patch sites in {path}, found {count}")
    path.write_text(text.replace(old, new))


def download_sdist(destination):
    metadata_url = f"https://pypi.org/pypi/{PACKAGE}/{VERSION}/json"
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        metadata = json.load(response)

    source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
    digest = hashlib.sha256()
    with urllib.request.urlopen(source["url"], timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)

    if digest.hexdigest() != source["digests"]["sha256"]:
        raise RuntimeError("Downloaded Metal Flash SDPA source failed its SHA-256 check")


def extract_safely(archive, destination):
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_resolved not in member_path.parents and member_path != destination_resolved:
                raise RuntimeError(f"Unsafe path in source archive: {member.name}")
        source.extractall(destination)


def patch_source(source):
    bridge = source / "csrc" / "mfa_bridge.mm"
    replace_exact(
        bridge,
        "#include <ATen/mps/MPSDevice.h>\n",
        "#include <ATen/mps/MPSDevice.h>\n#include <ATen/mps/MPSStream.h>\n",
    )
    replace_exact(
        bridge,
        "      @autoreleasepool {\n        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();",
        "      @autoreleasepool {\n        at::mps::getCurrentMPSStream()->endKernelCoalescing();\n        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();",
        expected_count=2,
    )
    # Keep MFA and the following PyTorch MPSGraph work on the same Metal
    # command buffer. PyTorch submits it when the downstream graph is encoded.
    replace_exact(bridge, "\n\n  torch::mps::commit();", "", expected_count=2)
    replace_exact(
        bridge,
        '#include "mfa/ccv_nnc_mfa_attention.hpp"\n',
        '#include "mfa/ccv_nnc_mfa_attention.hpp"\n\n'
        'void register_fused_ops(pybind11::module_& module);\n',
    )
    replace_exact(
        bridge,
        "PYBIND11_MODULE(_C, m) {\n",
        "PYBIND11_MODULE(_C, m) {\n  register_fused_ops(m);\n",
    )

    fused_source = Path(__file__).with_name("mps_fused_group_norm.mm")
    shutil.copyfile(fused_source, source / "csrc" / fused_source.name)

    setup = source / "setup.py"
    replace_exact(
        setup,
        "'cxx': ['-std=c++17', '-O2'],",
        "'cxx': ['-std=c++17', '-O2', '-Wno-invalid-specialization'],",
    )
    replace_exact(
        setup,
        "mm_sources = [\n    'csrc/mfa_bridge.mm',\n]",
        "mm_sources = [\n    'csrc/mfa_bridge.mm',\n"
        "    'csrc/mps_fused_group_norm.mm',\n]",
    )

    package_init = source / "metal_flash_sdpa" / "__init__.py"
    replace_exact(
        package_init,
        f'__version__ = "{VERSION}"\n',
        f'__version__ = "{VERSION}"\n'
        'A1111_MPS_STREAM_FIX = True\n'
        'A1111_MPS_DEFERRED_COMMIT = True\n'
        'A1111_MPS_FUSED_GROUP_NORM_SILU = True\n'
        'A1111_MPS_FUSED_GEGLU = True\n',
    )
    replace_exact(
        package_init,
        "from metal_flash_sdpa._C import mfa_attention_forward, mfa_attention_backward\n",
        "from metal_flash_sdpa._C import (\n"
        "    fused_geglu_forward,\n"
        "    fused_group_norm_silu_forward,\n"
        "    mfa_attention_backward,\n"
        "    mfa_attention_forward,\n"
        ")\n",
    )


def main():
    with tempfile.TemporaryDirectory(prefix="a1111-mps-flash-") as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / f"{PACKAGE}-{VERSION}.tar.gz"
        download_sdist(archive)
        extract_safely(archive, temporary_path)
        source = temporary_path / f"mps_flash_sdpa-{VERSION}"
        patch_source(source)
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
            str(source),
        ])


if __name__ == "__main__":
    main()
