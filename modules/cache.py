import json
import os
import os.path
import threading
import time

from modules.paths import data_path, script_path

cache_filename = os.environ.get('SD_WEBUI_CACHE_FILE', os.path.join(data_path, "cache.json"))
cache_data = None
cache_lock = threading.Lock()

dump_cache_after = None
dump_cache_thread = None


def dump_cache():
    """
    Marks cache for writing to disk. 5 seconds after no one else flags the cache for writing, it is written.
    """

    global dump_cache_after
    global dump_cache_thread

    def thread_func():
        global dump_cache_after
        global dump_cache_thread

        while dump_cache_after is not None and time.time() < dump_cache_after:
            time.sleep(1)

        with cache_lock:
            cache_filename_tmp = cache_filename + "-"
            with open(cache_filename_tmp, "w", encoding="utf8") as file:
                json.dump(cache_data, file, indent=4)

            os.replace(cache_filename_tmp, cache_filename)

            dump_cache_after = None
            dump_cache_thread = None

    with cache_lock:
        dump_cache_after = time.time() + 5
        if dump_cache_thread is None:
            dump_cache_thread = threading.Thread(name='cache-writer', target=thread_func)
            dump_cache_thread.start()


def cache(subsection):
    """
    Retrieves or initializes a cache for a specific subsection.

    Parameters:
        subsection (str): The subsection identifier for the cache.

    Returns:
        dict: The cache data for the specified subsection.
    """

    global cache_data

    if cache_data is None:
        with cache_lock:
            if cache_data is None:
                if not os.path.isfile(cache_filename):
                    cache_data = {}
                else:
                    try:
                        with open(cache_filename, "r", encoding="utf8") as file:
                            cache_data = json.load(file)
                    except Exception:
                        os.replace(cache_filename, os.path.join(script_path, "tmp", "cache.json"))
                        print('[ERROR] issue occurred while trying to read cache.json, move current cache to tmp/cache.json and create new cache')
                        cache_data = {}

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s


def cached_data_for_file(subsection, title, filename, func):
    """
    Retrieves or generates data for a specific file, using a caching mechanism.

    Parameters:
        subsection (str): The subsection of the cache to use.
        title (str): The title of the data entry in the subsection of the cache.
        filename (str): The path to the file to be checked for modifications.
        func (callable): A function that generates the data if it is not available in the cache.

    Returns:
        dict or None: The cached or generated data, or None if data generation fails.

    The `cached_data_for_file` function implements a caching mechanism for data stored in files.
    It checks if the data associated with the given `title` is present in the cache and compares the
    modification time of the file with the cached modification time. If the file has been modified,
    the cache is considered invalid and the data is regenerated using the provided `func`.
    Otherwise, the cached data is returned.

    If the data generation fails, None is returned to indicate the failure. Otherwise, the generated
    or cached data is returned as a dictionary.
    """

    existing_cache = cache(subsection)
    ondisk_mtime = os.path.getmtime(filename)

    entry = existing_cache.get(title)
    if entry:
        cached_mtime = entry.get("mtime", 0)
        if ondisk_mtime > cached_mtime:
            entry = None

    if not entry or 'value' not in entry:
        value = func()
        if value is None:
            return None

        entry = {'mtime': ondisk_mtime, 'value': value}
        existing_cache[title] = entry

        dump_cache()

    return entry['value']
