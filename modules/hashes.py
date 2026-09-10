import hashlib
import os.path

from modules import shared, errors
import modules.cache

dump_cache = modules.cache.dump_cache
cache = modules.cache.cache


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    try:
        ondisk_mtime = os.path.getmtime(filename)
    except FileNotFoundError:
        return None

    if title not in hashes:
        return None

    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)

    if ondisk_mtime != cached_mtime or cached_sha256 is None:
        return None

    return cached_sha256


def sha256(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value

    if shared.cmd_opts.no_hashing:
        return None

    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

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


def partial_hash_from_cache(filename, *, ignore_cache: bool = False, digits: int = 8):
    """old hash that only looks at a small part of the file and is prone to collisions
    kept for compatibility, don't use this for new things
    """
    try:
        filename = str(filename)
        mtime = os.path.getmtime(filename)
        hashes = cache('partial-hash')
        cache_entry = hashes.get(filename, {})
        cache_mtime = cache_entry.get("mtime", 0)
        cache_hash = cache_entry.get("hash", None)
        if mtime == cache_mtime and cache_hash and not ignore_cache:
            return cache_hash[0:digits]

        with open(filename, 'rb') as file:
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            partial_hash = m.hexdigest()
            hashes[filename] = {'mtime': mtime, 'hash': partial_hash}
            return partial_hash[0:digits]

    except FileNotFoundError:
        pass
    except Exception:
        errors.report(f'Error calculating partial hash for {filename}', exc_info=True)
    return 'NOFILE'
