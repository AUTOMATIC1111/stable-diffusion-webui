import configparser
import functools
import os
import threading
import re

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions = []

os.makedirs(extensions_dir, exist_ok=True)


def active():
    if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions == "all":
        return []
    elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions == "extra":
        return [x for x in extensions if x.enabled and x.is_builtin]
    else:
        return [x for x in extensions if x.enabled]


class Extension:
    lock = threading.Lock()
    cached_fields = ['remote', 'commit_date', 'branch', 'commit_hash', 'version']

    def __init__(self, name, path, enabled=True, is_builtin=False, canonical_name=None):
        self.name = name
        self.canonical_name = canonical_name or name.lower()
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

    @functools.cached_property
    def metadata(self):
        if os.path.isfile(os.path.join(self.path, "sd_webui_metadata.ini")):
            try:
                config = configparser.ConfigParser()
                config.read(os.path.join(self.path, "sd_webui_metadata.ini"))
                return config
            except Exception:
                errors.report(f"Error reading sd_webui_metadata.ini for extension {self.canonical_name}.",
                              exc_info=True)
        return None

    def to_dict(self):
        return {x: getattr(self, x) for x in self.cached_fields}

    def from_dict(self, d):
        for field in self.cached_fields:
            setattr(self, field, d[field])

    def read_info_from_repo(self):
        if self.is_builtin or self.have_info_from_repo:
            return

        def read_from_repo():
            with self.lock:
                if self.have_info_from_repo:
                    return

                self.do_read_info_from_repo()

                return self.to_dict()

        try:
            d = cache.cached_data_for_file('extensions-git', self.name, os.path.join(self.path, ".git"), read_from_repo)
            self.from_dict(d)
        except FileNotFoundError:
            pass
        self.status = 'unknown' if self.status == '' else self.status

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

    if shared.cmd_opts.disable_all_extensions:
        print("*** \"--disable-all-extensions\" arg was used, will not load any extensions ***")
    elif shared.opts.disable_all_extensions == "all":
        print("*** \"Disable all extensions\" option was set, will not load any extensions ***")
    elif shared.cmd_opts.disable_extra_extensions:
        print("*** \"--disable-extra-extensions\" arg was used, will only load built-in extensions ***")
    elif shared.opts.disable_all_extensions == "extra":
        print("*** \"Disable all extensions\" option was set, will only load built-in extensions ***")

    extension_dependency_map = {}

    # scan through extensions directory and load metadata
    for dirname in [extensions_builtin_dir, extensions_dir]:
        if not os.path.isdir(dirname):
            continue

        for extension_dirname in sorted(os.listdir(dirname)):
            path = os.path.join(dirname, extension_dirname)
            if not os.path.isdir(path):
                continue

            canonical_name = extension_dirname
            requires = None

            if os.path.isfile(os.path.join(path, "sd_webui_metadata.ini")):
                try:
                    config = configparser.ConfigParser()
                    config.read(os.path.join(path, "sd_webui_metadata.ini"))
                    canonical_name = config.get("Extension", "Name", fallback=canonical_name)
                    requires = config.get("Extension", "Requires", fallback=None)
                except Exception:
                    errors.report(f"Error reading sd_webui_metadata.ini for extension {extension_dirname}. "
                                  f"Will load regardless.", exc_info=True)

            canonical_name = canonical_name.lower().strip()

            # check for duplicated canonical names
            if canonical_name in extension_dependency_map:
                errors.report(f"Duplicate canonical name \"{canonical_name}\" found in extensions "
                              f"\"{extension_dirname}\" and \"{extension_dependency_map[canonical_name]['dirname']}\". "
                              f"The current loading extension will be discarded.", exc_info=False)
                continue

            # both "," and " " are accepted as separator
            requires = list(filter(None, re.split(r"[,\s]+", requires.lower()))) if requires else []

            extension_dependency_map[canonical_name] = {
                "dirname": extension_dirname,
                "path": path,
                "requires": requires,
            }

    # check for requirements
    for (_, extension_data) in extension_dependency_map.items():
        dirname, path, requires = extension_data['dirname'], extension_data['path'], extension_data['requires']
        requirement_met = True
        for req in requires:
            if req not in extension_dependency_map:
                errors.report(f"Extension \"{dirname}\" requires \"{req}\" which is not installed. "
                              f"The current loading extension will be discarded.", exc_info=False)
                requirement_met = False
                break
            dep_dirname = extension_dependency_map[req]['dirname']
            if dep_dirname in shared.opts.disabled_extensions:
                errors.report(f"Extension \"{dirname}\" requires \"{dep_dirname}\" which is disabled. "
                              f"The current loading extension will be discarded.", exc_info=False)
                requirement_met = False
                break

        is_builtin = dirname == extensions_builtin_dir
        extension = Extension(name=dirname, path=path,
                              enabled=dirname not in shared.opts.disabled_extensions and requirement_met,
                              is_builtin=is_builtin)
        extensions.append(extension)
