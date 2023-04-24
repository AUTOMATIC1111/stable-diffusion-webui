import os
import time
import git

from modules import shared, errors
from modules.paths_internal import extensions_dir, extensions_builtin_dir

extensions = []

if not os.path.exists(extensions_dir):
    os.makedirs(extensions_dir)


def active():
    if shared.opts.disable_all_extensions == "all":
        return []
    elif shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
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
        self.remote = None
        self.have_info_from_repo = False

    def read_info_from_repo(self):
        if self.have_info_from_repo:
            return

        self.have_info_from_repo = True

        repo = None
        try:
            if os.path.exists(os.path.join(self.path, ".git")):
                repo = git.Repo(self.path)
        except Exception as e:
            errors.display(e, f'github info from {self.path}')

        if repo is None or repo.bare:
            self.remote = None
        else:
            try:
                self.status = 'unknown'
                self.remote = next(repo.remote().urls, None)
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
            priority = '50'
            if os.path.isfile(os.path.join(dirpath, "..", ".priority")):
                with open(os.path.join(dirpath, "..", ".priority"), "r", encoding="utf-8") as f:
                    priority = str(f.read().strip())
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename), priority))

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

    if not os.path.isdir(extensions_dir):
        return

    if shared.opts.disable_all_extensions == "all" or shared.opts.disable_all_extensions == "extra":
        shared.log.warning("Option set: Disable all extensions")

    extension_paths = []
    extension_names = []
    for dirname in [extensions_builtin_dir, extensions_dir]:
        if not os.path.isdir(dirname):
            return

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue
            if extension_dirname in extension_names:
                shared.log.info(f'Skipping conflicting extension: {path}')
                continue
            extension_names.append(extension_dirname)
            extension_paths.append((extension_dirname, path, dirname == extensions_builtin_dir))

    for dirname, path, is_builtin in extension_paths:
        extension = Extension(name=dirname, path=path, enabled=dirname not in shared.opts.disabled_extensions, is_builtin=is_builtin)
        extensions.append(extension)
