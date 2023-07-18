import hashlib
import os.path
from rich import progress
from modules import shared
from modules.paths import data_path

cache_filename = os.path.join(data_path, "cache.json")
cache_data = None


def dump_cache():
    shared.writefile(cache_data, cache_filename)


def cache(subsection):
    global cache_data # pylint: disable=global-statement
    if cache_data is None:
        if not os.path.isfile(cache_filename):
            cache_data = {}
        else:
            cache_data = shared.readfile(cache_filename)
    s = cache_data.get(subsection, {})
    cache_data[subsection] = s
    return s


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    with progress.open(filename, 'rb', description=f'Calculating model hash: [cyan]{filename}', auto_refresh=True) as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)
    if title not in hashes:
        return None
    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)
    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None
    return cached_sha256


def sha256(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value
    if shared.cmd_opts.no_hashing:
        return None
    if use_addnet_hash:
        with progress.open(filename, 'rb', description=f'Calculating model hash: [cyan]{filename}', auto_refresh=True) as f:
            sha256_value = addnet_hash_safetensors(f)
    else:
        sha256_value = calculate_sha256(filename)
    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }
    dump_cache()
    return sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")
    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)
    return hash_sha256.hexdigest()
