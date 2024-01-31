# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest import mock
import torch
from torch import nn

# need some explicit imports due to https://github.com/pytorch/pytorch/issues/38964
import annotator.oneformer.detectron2  # noqa F401
from annotator.oneformer.detectron2.structures import Boxes, Instances
from annotator.oneformer.detectron2.utils.env import _import_file

_counter = 0


def _clear_jit_cache():
    from torch.jit._recursive import concrete_type_store
    from torch.jit._state import _jit_caching_layer

    concrete_type_store.type_store.clear()  # for modules
    _jit_caching_layer.clear()  # for free functions


def _add_instances_conversion_methods(newInstances):
    """
    Add from_instances methods to the scripted Instances class.
    """
    cls_name = newInstances.__name__

    @torch.jit.unused
    def from_instances(instances: Instances):
        """
        Create scripted Instances from original Instances
        """
        fields = instances.get_fields()
        image_size = instances.image_size
        ret = newInstances(image_size)
        for name, val in fields.items():
            assert hasattr(ret, f"_{name}"), f"No attribute named {name} in {cls_name}"
            setattr(ret, name, deepcopy(val))
        return ret

    newInstances.from_instances = from_instances


@contextmanager
def patch_instances(fields):
    """
    A contextmanager, under which the Instances class in detectron2 is replaced
    by a statically-typed scriptable class, defined by `fields`.
    See more in `scripting_with_instances`.
    """

    with tempfile.TemporaryDirectory(prefix="detectron2") as dir, tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", dir=dir, delete=False
    ) as f:
        try:
            # Objects that use Instances should not reuse previously-compiled
            # results in cache, because `Instances` could be a new class each time.
            _clear_jit_cache()

            cls_name, s = _gen_instance_module(fields)
            f.write(s)
            f.flush()
            f.close()

            module = _import(f.name)
            new_instances = getattr(module, cls_name)
            _ = torch.jit.script(new_instances)
            # let torchscript think Instances was scripted already
            Instances.__torch_script_class__ = True
            # let torchscript find new_instances when looking for the jit type of Instances
            Instances._jit_override_qualname = torch._jit_internal._qualified_name(new_instances)

            _add_instances_conversion_methods(new_instances)
            yield new_instances
        finally:
            try:
                del Instances.__torch_script_class__
                del Instances._jit_override_qualname
            except AttributeError:
                pass
            sys.modules.pop(module.__name__)


def _gen_instance_class(fields):
    """
    Args:
        fields (dict[name: type])
    """

    class _FieldType:
        def __init__(self, name, type_):
            assert isinstance(name, str), f"Field name must be str, got {name}"
            self.name = name
            self.type_ = type_
            self.annotation = f"{type_.__module__}.{type_.__name__}"

    fields = [_FieldType(k, v) for k, v in fields.items()]

    def indent(level, s):
        return " " * 4 * level + s

    lines = []

    global _counter
    _counter += 1

    cls_name = "ScriptedInstances{}".format(_counter)

    field_names = tuple(x.name for x in fields)
    extra_args = ", ".join([f"{f.name}: Optional[{f.annotation}] = None" for f in fields])
    lines.append(
        f"""
class {cls_name}:
    def __init__(self, image_size: Tuple[int, int], {extra_args}):
        self.image_size = image_size
        self._field_names = {field_names}
"""
    )

    for f in fields:
        lines.append(
            indent(2, f"self._{f.name} = torch.jit.annotate(Optional[{f.annotation}], {f.name})")
        )

    for f in fields:
        lines.append(
            f"""
    @property
    def {f.name}(self) -> {f.annotation}:
        # has to use a local for type refinement
        # https://pytorch.org/docs/stable/jit_language_reference.html#optional-type-refinement
        t = self._{f.name}
        assert t is not None, "{f.name} is None and cannot be accessed!"
        return t

    @{f.name}.setter
    def {f.name}(self, value: {f.annotation}) -> None:
        self._{f.name} = value
"""
        )

    # support method `__len__`
    lines.append(
        """
    def __len__(self) -> int:
"""
    )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            return len(t)
"""
        )
    lines.append(
        """
        raise NotImplementedError("Empty Instances does not support __len__!")
"""
    )

    # support method `has`
    lines.append(
        """
    def has(self, name: str) -> bool:
"""
    )
    for f in fields:
        lines.append(
            f"""
        if name == "{f.name}":
            return self._{f.name} is not None
"""
        )
    lines.append(
        """
        return False
"""
    )

    # support method `to`
    none_args = ", None" * len(fields)
    lines.append(
        f"""
    def to(self, device: torch.device) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
    )
    for f in fields:
        if hasattr(f.type_, "to"):
            lines.append(
                f"""
        t = self._{f.name}
        if t is not None:
            ret._{f.name} = t.to(device)
"""
            )
        else:
            # For now, ignore fields that cannot be moved to devices.
            # Maybe can support other tensor-like classes (e.g. __torch_function__)
            pass
    lines.append(
        """
        return ret
"""
    )

    # support method `getitem`
    none_args = ", None" * len(fields)
    lines.append(
        f"""
    def __getitem__(self, item) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
    )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            ret._{f.name} = t[item]
"""
        )
    lines.append(
        """
        return ret
"""
    )

    # support method `cat`
    # this version does not contain checks that all instances have same size and fields
    none_args = ", None" * len(fields)
    lines.append(
        f"""
    def cat(self, instances: List["{cls_name}"]) -> "{cls_name}":
        ret = {cls_name}(self.image_size{none_args})
"""
    )
    for f in fields:
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            values: List[{f.annotation}] = [x.{f.name} for x in instances]
            if torch.jit.isinstance(t, torch.Tensor):
                ret._{f.name} = torch.cat(values, dim=0)
            else:
                ret._{f.name} = t.cat(values)
"""
        )
    lines.append(
        """
        return ret"""
    )

    # support method `get_fields()`
    lines.append(
        """
    def get_fields(self) -> Dict[str, Tensor]:
        ret = {}
    """
    )
    for f in fields:
        if f.type_ == Boxes:
            stmt = "t.tensor"
        elif f.type_ == torch.Tensor:
            stmt = "t"
        else:
            stmt = f'assert False, "unsupported type {str(f.type_)}"'
        lines.append(
            f"""
        t = self._{f.name}
        if t is not None:
            ret["{f.name}"] = {stmt}
        """
        )
    lines.append(
        """
        return ret"""
    )
    return cls_name, os.linesep.join(lines)


def _gen_instance_module(fields):
    # TODO: find a more automatic way to enable import of other classes
    s = """
from copy import deepcopy
import torch
from torch import Tensor
import typing
from typing import *

import annotator.oneformer.detectron2
from annotator.oneformer.detectron2.structures import Boxes, Instances

"""

    cls_name, cls_def = _gen_instance_class(fields)
    s += cls_def
    return cls_name, s


def _import(path):
    return _import_file(
        "{}{}".format(sys.modules[__name__].__name__, _counter), path, make_importable=True
    )


@contextmanager
def patch_builtin_len(modules=()):
    """
    Patch the builtin len() function of a few detectron2 modules
    to use __len__ instead, because __len__ does not convert values to
    integers and therefore is friendly to tracing.

    Args:
        modules (list[stsr]): names of extra modules to patch len(), in
            addition to those in detectron2.
    """

    def _new_len(obj):
        return obj.__len__()

    with ExitStack() as stack:
        MODULES = [
            "detectron2.modeling.roi_heads.fast_rcnn",
            "detectron2.modeling.roi_heads.mask_head",
            "detectron2.modeling.roi_heads.keypoint_head",
        ] + list(modules)
        ctxs = [stack.enter_context(mock.patch(mod + ".len")) for mod in MODULES]
        for m in ctxs:
            m.side_effect = _new_len
        yield


def patch_nonscriptable_classes():
    """
    Apply patches on a few nonscriptable detectron2 classes.
    Should not have side-effects on eager usage.
    """
    # __prepare_scriptable__ can also be added to models for easier maintenance.
    # But it complicates the clean model code.

    from annotator.oneformer.detectron2.modeling.backbone import ResNet, FPN

    # Due to https://github.com/pytorch/pytorch/issues/36061,
    # we change backbone to use ModuleList for scripting.
    # (note: this changes param names in state_dict)

    def prepare_resnet(self):
        ret = deepcopy(self)
        ret.stages = nn.ModuleList(ret.stages)
        for k in self.stage_names:
            delattr(ret, k)
        return ret

    ResNet.__prepare_scriptable__ = prepare_resnet

    def prepare_fpn(self):
        ret = deepcopy(self)
        ret.lateral_convs = nn.ModuleList(ret.lateral_convs)
        ret.output_convs = nn.ModuleList(ret.output_convs)
        for name, _ in self.named_children():
            if name.startswith("fpn_"):
                delattr(ret, name)
        return ret

    FPN.__prepare_scriptable__ = prepare_fpn

    # Annotate some attributes to be constants for the purpose of scripting,
    # even though they are not constants in eager mode.
    from annotator.oneformer.detectron2.modeling.roi_heads import StandardROIHeads

    if hasattr(StandardROIHeads, "__annotations__"):
        # copy first to avoid editing annotations of base class
        StandardROIHeads.__annotations__ = deepcopy(StandardROIHeads.__annotations__)
        StandardROIHeads.__annotations__["mask_on"] = torch.jit.Final[bool]
        StandardROIHeads.__annotations__["keypoint_on"] = torch.jit.Final[bool]


# These patches are not supposed to have side-effects.
patch_nonscriptable_classes()


@contextmanager
def freeze_training_mode(model):
    """
    A context manager that annotates the "training" attribute of every submodule
    to constant, so that the training codepath in these modules can be
    meta-compiled away. Upon exiting, the annotations are reverted.
    """
    classes = {type(x) for x in model.modules()}
    # __constants__ is the old way to annotate constants and not compatible
    # with __annotations__ .
    classes = {x for x in classes if not hasattr(x, "__constants__")}
    for cls in classes:
        cls.__annotations__["training"] = torch.jit.Final[bool]
    yield
    for cls in classes:
        cls.__annotations__["training"] = bool
