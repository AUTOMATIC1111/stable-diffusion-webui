from __future__ import annotations

import io
import subprocess

import git


class Git(git.Git):
    """
    Git subclassed to never use persistent processes.
    """

    def _get_persistent_cmd(self, attr_name, cmd_name, *args, **kwargs):
        raise NotImplementedError(f"Refusing to use persistent process: {attr_name} ({cmd_name} {args} {kwargs})")

    def get_object_header(self, ref: str | bytes) -> tuple[str, str, int]:
        ret = subprocess.check_output(
            [self.GIT_PYTHON_GIT_EXECUTABLE, "cat-file", "--batch-check"],
            input=self._prepare_ref(ref),
            cwd=self._working_dir,
            timeout=2,
        )
        return self._parse_object_header(ret)

    def stream_object_data(self, ref: str) -> tuple[str, str, int, "Git.CatFileContentStream"]:
        # Not really streaming, per se; this buffers the entire object in memory.
        # Shouldn't be a problem for our use case, since we're only using this for
        # object headers (commit objects).
        ret = subprocess.check_output(
            [self.GIT_PYTHON_GIT_EXECUTABLE, "cat-file", "--batch"],
            input=self._prepare_ref(ref),
            cwd=self._working_dir,
            timeout=30,
        )
        bio = io.BytesIO(ret)
        hexsha, typename, size = self._parse_object_header(bio.readline())
        return (hexsha, typename, size, self.CatFileContentStream(size, bio))


class Repo(git.Repo):
    GitCommandWrapperType = Git
