import os
import sys
import traceback

import git

from modules import paths, shared

extensions = []
extensions_dir = os.path.join(paths.script_path, "extensions")


def active():
    return [x for x in extensions if x.enabled]


class Extension:
    def __init__(self, name, path, enabled=True):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False

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
        for fetch in repo.remote().fetch("--dry-run"):
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "behind"
                return

        self.can_update = False
        self.status = "latest"

    def pull(self):
        repo = git.Repo(self.path)
        repo.remotes.origin.pull()


def list_extensions():
    extensions.clear()

    if not os.path.isdir(extensions_dir):
        return

    for dirname in sorted(os.listdir(extensions_dir)):
        path = os.path.join(extensions_dir, dirname)
        if not os.path.isdir(path):
            continue

        extension = Extension(name=dirname, path=path, enabled=dirname not in shared.opts.disabled_extensions)
        extensions.append(extension)

