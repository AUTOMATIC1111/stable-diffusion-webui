import os
import threading

from modules import shared, errors
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions = []

os.makedirs(extensions_dir, exist_ok=True)


def active():
    if shared.opts.disable_all_extensions == "all":
        return []
    elif shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class Extension:
    lock = threading.Lock()

    def __init__(self, name, path, enabled=True, is_builtin=False):
        self.name = name
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        with self.lock:
            if self.have_info_from_repo:
                return

            self.do_read_info_from_repo()

    def do_read_info_from_repo(self):
        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = Repo(self.path)
        except Exception:
            errors.report(f"Error reading github repository info from {self.path}", exc_info=True)

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.status = 'unknown'
                self.remote = next(repo.remote().urls, None)
                commit = repo.head.commit
                self.commit_date = commit.committed_date
                if repo.active_branch:
                    self.branch = repo.active_branch.name
                self.commit_hash = commit.hexsha
                self.version = self.commit_hash[:8]

            except Exception:
                errors.report(f"Failed reading extension data from Git repository ({self.name})", exc_info=True)
                self.remote = None

        self.have_info_from_repo = True

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
        repo = Repo(self.path)
        for fetch in repo.remote().fetch(dry_run=True):
            if fetch.flags != fetch.HEAD_UPTODATE:
                self.can_update = True
                self.status = "new commits"
                return

        try:
            origin = repo.rev_parse('origin')
            if repo.head.commit != origin:
                self.can_update = True
                self.status = "behind HEAD"
                return
        except Exception:
            self.can_update = False
            self.status = "unknown (remote error)"
            return

        self.can_update = False
        self.status = "latest"

    def fetch_and_reset_hard(self, commit='origin'):
        repo = Repo(self.path)
        # Fix: `error: Your local changes to the following files would be overwritten by merge`,
        # because WSL2 Docker set 755 file permissions instead of 644, this results to the error.
        repo.git.fetch(all=True)
        repo.git.reset(commit, hard=True)
        self.have_info_from_repo = False


def list_extensions():
    extensions.clear()

    if not os.path.isdir(extensions_dir):
        return

    if shared.opts.disable_all_extensions == "all":
        print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
    elif shared.opts.disable_all_extensions == "extra":
        print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

    extension_paths = []
    for dirname in [extensions_dir, extensions_builtin_dir]:
        if not os.path.isdir(dirname):
            return

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            extension_paths.append((extension_dirname, path, dirname == extensions_builtin_dir))

    for dirname, path, is_builtin in extension_paths:
        extension = Extension(name=dirname, path=path, enabled=dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin)
        extensions.append(extension)
