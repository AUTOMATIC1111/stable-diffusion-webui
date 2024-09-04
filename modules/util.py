import os
import re

from modules import shared
from modules.paths_internal import script_path, cwd


def natural_sort_key(s, regex=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in regex.split(s)]


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=natural_sort_key) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def html_path(filename):
    return os.path.join(script_path, "html", filename)


def html(filename):
    path = html_path(filename)

    try:
        with open(path, encoding="utf8") as file:
            return file.read()
    except OSError:
        return ""


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    items = list(os.walk(path, followlinks=True))
    items = sorted(items, key=lambda x: natural_sort_key(x[0]))

    for root, _, files in items:
        for filename in sorted(files, key=natural_sort_key):
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in allowed_extensions:
                    continue

            if not shared.opts.list_hidden_files and ("/." in root or "\\." in root):
                continue

            yield os.path.join(root, filename)


def ldm_print(*args, **kwargs):
    if shared.opts.hide_ldm_prints:
        return

    print(*args, **kwargs)


def truncate_path(target_path, base_path=cwd):
    abs_target, abs_base = os.path.abspath(target_path), os.path.abspath(base_path)
    try:
        if os.path.commonpath([abs_target, abs_base]) == abs_base:
            return os.path.relpath(abs_target, abs_base)
    except ValueError:
        pass
    return abs_target


class MassFileListerCachedDir:
    """A class that caches file metadata for a specific directory."""

    def __init__(self, dirname):
        self.files = None
        self.files_cased = None
        self.dirname = dirname

        stats = ((x.name, x.stat(follow_symlinks=False)) for x in os.scandir(self.dirname))
        files = [(n, s.st_mtime, s.st_ctime) for n, s in stats]
        self.files = {x[0].lower(): x for x in files}
        self.files_cased = {x[0]: x for x in files}

    def update_entry(self, filename):
        """Add a file to the cache"""
        file_path = os.path.join(self.dirname, filename)
        try:
            stat = os.stat(file_path)
            entry = (filename, stat.st_mtime, stat.st_ctime)
            self.files[filename.lower()] = entry
            self.files_cased[filename] = entry
        except FileNotFoundError as e:
            print(f'MassFileListerCachedDir.add_entry: "{file_path}" {e}')


class MassFileLister:
    """A class that provides a way to check for the existence and mtime/ctile of files without doing more than one stat call per file."""

    def __init__(self):
        self.cached_dirs = {}

    def find(self, path):
        """
        Find the metadata for a file at the given path.

        Returns:
            tuple or None: A tuple of (name, mtime, ctime) if the file exists, or None if it does not.
        """

        dirname, filename = os.path.split(path)

        cached_dir = self.cached_dirs.get(dirname)
        if cached_dir is None:
            cached_dir = MassFileListerCachedDir(dirname)
            self.cached_dirs[dirname] = cached_dir

        stats = cached_dir.files_cased.get(filename)
        if stats is not None:
            return stats

        stats = cached_dir.files.get(filename.lower())
        if stats is None:
            return None

        try:
            os_stats = os.stat(path, follow_symlinks=False)
            return filename, os_stats.st_mtime, os_stats.st_ctime
        except Exception:
            return None

    def exists(self, path):
        """Check if a file exists at the given path."""

        return self.find(path) is not None

    def mctime(self, path):
        """
        Get the modification and creation times for a file at the given path.

        Returns:
            tuple: A tuple of (mtime, ctime) if the file exists, or (0, 0) if it does not.
        """

        stats = self.find(path)
        return (0, 0) if stats is None else stats[1:3]

    def reset(self):
        """Clear the cache of all directories."""
        self.cached_dirs.clear()

    def update_file_entry(self, path):
        """Update the cache for a specific directory."""
        dirname, filename = os.path.split(path)
        if cached_dir := self.cached_dirs.get(dirname):
            cached_dir.update_entry(filename)

def topological_sort(dependencies):
    """Accepts a dictionary mapping name to its dependencies, returns a list of names ordered according to dependencies.
    Ignores errors relating to missing dependencies or circular dependencies
    """

    visited = {}
    result = []

    def inner(name):
        visited[name] = True

        for dep in dependencies.get(name, []):
            if dep in dependencies and dep not in visited:
                inner(dep)

        result.append(name)

    for depname in dependencies:
        if depname not in visited:
            inner(depname)

    return result


def open_folder(path):
    """Open a folder in the file manager of the respect OS."""
    # import at function level to avoid potential issues
    import gradio as gr
    import platform
    import sys
    import subprocess

    if not os.path.exists(path):
        msg = f'Folder "{path}" does not exist. after you save an image, the folder will be created.'
        print(msg)
        gr.Info(msg)
        return
    elif not os.path.isdir(path):
        msg = f"""
WARNING
An open_folder request was made with an path that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {path}
"""
        print(msg, file=sys.stderr)
        gr.Warning(msg)
        return

    path = os.path.normpath(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        subprocess.Popen(["explorer.exe", subprocess.check_output(["wslpath", "-w", path])])
    else:
        subprocess.Popen(["xdg-open", path])
