import os
from datetime import datetime
import git
from modules import shared, errors
from modules.paths import extensions_dir, extensions_builtin_dir


extensions = []


if not os.path.exists(extensions_dir):
    os.makedirs(extensions_dir)


def active():
    if shared.opts.disable_all_extensions == "all":
        return []
    elif shared.opts.disable_all_extensions == "user":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class Extension:
    def __init__(self, name, path, enabled=True, is_builtin=False):
        self.name = name
        self.git_name = ''
        self.path = path
        self.enabled = enabled
        self.status = ''
        self.can_update = False
        self.is_builtin = is_builtin
        self.commit_hash = ''
        self.commit_date = None
        self.version = ''
        self.description = ''
        self.branch = None
        self.remote = None
        self.have_info_from_repo = False
        self.mtime = 0
        self.ctime = 0

    def read_info(self, force=False):
        if self.have_info_from_repo and not force:
            return
        self.have_info_from_repo = True
        repo = None
        self.mtime = datetime.fromtimestamp(os.path.getmtime(self.path)).isoformat() + 'Z'
        self.ctime = datetime.fromtimestamp(os.path.getctime(self.path)).isoformat() + 'Z'
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
                if len(repo.remotes) == 0:
                    shared.log.debug(f"Extension: no remotes info repo={self.name}")
                    return
                self.git_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]
                self.description = repo.description
                if self.description is None or self.description.startswith("Unnamed repository"):
                    self.description = "[No description]"
                self.remote = next(repo.remote().urls, None)
                head = repo.head.commit
                self.commit_date = repo.head.commit.committed_date
                try:
                    if repo.active_branch:
                        self.branch = repo.active_branch.name
                except Exception:
                    pass
                self.commit_hash = head.hexsha
                self.version = f"<p>{self.commit_hash[:8]}</p><p>{datetime.fromtimestamp(self.commit_date).strftime('%a %b%d %Y %H:%M')}</p>"
            except Exception as ex:
                shared.log.error(f"Extension: failed reading data from git repo={self.name}: {ex}")
                self.remote = None

    def list_files(self, subdir, extension):
        from modules import scripts
        dirpath = os.path.join(self.path, subdir)
        if not os.path.isdir(dirpath):
            return []
        res = []
        for filename in sorted(os.listdir(dirpath)):
            if not filename.endswith(".py") and not filename.endswith(".js") and not filename.endswith(".mjs"):
                continue
            priority = '50'
            if os.path.isfile(os.path.join(dirpath, "..", ".priority")):
                with open(os.path.join(dirpath, "..", ".priority"), "r", encoding="utf-8") as f:
                    priority = str(f.read().strip())
            res.append(scripts.ScriptFile(self.path, filename, os.path.join(dirpath, filename), priority))
            if priority != '50':
                shared.log.debug(f'Extension priority override: {os.path.dirname(dirpath)}:{priority}')
        res = [x for x in res if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]
        return res

    def check_updates(self):
        try:
            repo = git.Repo(self.path)
        except Exception:
            self.can_update = False
            return
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

    def git_fetch(self, commit='origin'):
        repo = git.Repo(self.path)
        # Fix: `error: Your local changes to the following files would be overwritten by merge`,
        # because WSL2 Docker set 755 file permissions instead of 644, this results to the error.
        repo.git.fetch(all=True)
        repo.git.reset('origin', hard=True)
        repo.git.reset(commit, hard=True)
        self.have_info_from_repo = False


def list_extensions():
    extensions.clear()
    if not os.path.isdir(extensions_dir):
        return
    if shared.opts.disable_all_extensions == "all" or shared.opts.disable_all_extensions == "user":
        shared.log.warning(f"Option set: Disable extensions: {shared.opts.disable_all_extensions}")
    extension_paths = []
    extension_names = []
    extension_folders = [extensions_builtin_dir] if shared.cmd_opts.safe else [extensions_builtin_dir, extensions_dir]
    for dirname in extension_folders:
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
    disabled_extensions = shared.opts.disabled_extensions + shared.temp_disable_extensions()
    for dirname, path, is_builtin in extension_paths:
        extension = Extension(name=dirname, path=path, enabled=dirname not in disabled_extensions, is_builtin=is_builtin)
        extensions.append(extension)
    shared.log.info(f'Disabled extensions: {[e.name for e in extensions if not e.enabled]}')
