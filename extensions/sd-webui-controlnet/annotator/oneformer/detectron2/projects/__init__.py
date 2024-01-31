# Copyright (c) Facebook, Inc. and its affiliates.
import importlib.abc
import importlib.util
from pathlib import Path

__all__ = []

_PROJECTS = {
    "point_rend": "PointRend",
    "deeplab": "DeepLab",
    "panoptic_deeplab": "Panoptic-DeepLab",
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "projects"

if _PROJECT_ROOT.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _D2ProjectsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("detectron2.projects."):
                return
            project_name = name.split(".")[-1]
            project_dir = _PROJECTS.get(project_name)
            if not project_dir:
                return
            target_file = _PROJECT_ROOT / f"{project_dir}/{project_name}/__init__.py"
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(name, target_file)

    import sys

    sys.meta_path.append(_D2ProjectsFinder())
