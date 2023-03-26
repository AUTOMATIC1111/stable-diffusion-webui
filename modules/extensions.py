import os
import sys
import traceback

import time
import git

from modules import paths, shared
from modules.paths_internal import extensions_dir, extensions_builtin_dir

extensions = []

if not os.path.exists(paths.extensions_dir):
    os.makedirs(paths.extensions_dir)

def active():
    return [x for x in extensions if x.enabled]


class Extension:
    def __init__(self, name, path, enabled=True, is_builtin=False):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.version = ''

        repo = None
        try:
            if os.path.exists(os.path.join(path, ".git")):
                repo = git.Repo(path)
        except Exception:
            print(f"Error reading github repository info from {path}:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.remote = next(repo.remote().urls, None)
                self.status = 'unknown'
                head = repo.head.commit
                ts = time.asctime(time.gmtime(repo.head.commit.committed_date))
                self.version = f'{head.hexsha[:8]} ({ts})'

            except Exception:
                self.remote = None

    def list_files(self, subdir, extension):
        from modules import scripts

        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []

        res = []
        for filename in sorted(os.listdir(dirpath)):
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename)))

        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

        return res

    def check_updates(self):
        repo = git.Repo(self.path)
        for fetch in repo.remote().fetch(dry_run=True):
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "behind"
                return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self):
        repo = git.Repo(self.path)
        # Fix: `error: Your local changes to the following files would be overwritten by merge`,
        # because WSL2 Docker set 755 file permissions instead of 644, this results to the error.
        repo.git.fetch(all=True)
        repo.git.reset('origin', hard=True)


def list_extensions():
    extensions.clear()

    if not os.path.isdir(paths.extensions_dir):
        return

    extension_paths = []
    for dirname in [paths.extensions_dir, paths.extensions_builtin_dir]:
        if not os.path.isdir(dirname):
            return

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            extension_paths.append((extension_dirname, path, dirname == paths.extensions_builtin_dir))

    for dirname, path, is_builtin in extension_paths:
        extension = Extension(name=dirname, path=path, enabled=dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin)
        extensions.append(extension)

