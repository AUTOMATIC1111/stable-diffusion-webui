# Copyright (c) Facebook, Inc. and its affiliates.

import os
import torch

from annotator.oneformer.detectron2.utils.file_io import PathManager

from .torchscript_patch import freeze_training_mode, patch_instances

__all__ = ["scripting_with_instances", "dump_torchscript_IR"]


def scripting_with_instances(model, fields):
    """
    Run :func:`torch.jit.script` on a model that uses the :class:`Instances` class. Since
    attributes of :class:`Instances` are "dynamically" added in eager mode，it is difficult
    for scripting to support it out of the box. This function is made to support scripting
    a model that uses :class:`Instances`. It does the following:

    1. Create a scriptable ``new_Instances`` class which behaves similarly to ``Instances``,
       but with all attributes been "static".
       The attributes need to be statically declared in the ``fields`` argument.
    2. Register ``new_Instances``, and force scripting compiler to
       use it when trying to compile ``Instances``.

    After this function, the process will be reverted. User should be able to script another model
    using different fields.

    Example:
        Assume that ``Instances`` in the model consist of two attributes named
        ``proposal_boxes`` and ``objectness_logits`` with type :class:`Boxes` and
        :class:`Tensor` respectively during inference. You can call this function like:
        ::
            fields = {"proposal_boxes": Boxes, "objectness_logits": torch.Tensor}
            torchscipt_model =  scripting_with_instances(model, fields)

    Note:
        It only support models in evaluation mode.

    Args:
        model (nn.Module): The input model to be exported by scripting.
        fields (Dict[str, type]): Attribute names and corresponding type that
            ``Instances`` will use in the model. Note that all attributes used in ``Instances``
            need to be added, regardless of whether they are inputs/outputs of the model.
            Data type not defined in detectron2 is not supported for now.

    Returns:
        torch.jit.ScriptModule: the model in torchscript format
    """
    assert (
        not model.training
    ), "Currently we only support exporting models in evaluation mode to torchscript"

    with freeze_training_mode(model), patch_instances(fields):
        scripted_model = torch.jit.script(model)
        return scripted_model


# alias for old name
export_torchscript_with_instances = scripting_with_instances


def dump_torchscript_IR(model, dir):
    """
    Dump IR of a TracedModule/ScriptModule/Function in various format (code, graph,
    inlined graph). Useful for debugging.

    Args:
        model (TracedModule/ScriptModule/ScriptFUnction): traced or scripted module
        dir (str): output directory to dump files.
    """
    dir = os.path.expanduser(dir)
    PathManager.mkdirs(dir)

    def _get_script_mod(mod):
        if isinstance(mod, torch.jit.TracedModule):
            return mod._actual_script_module
        return mod

    # Dump pretty-printed code: https://pytorch.org/docs/stable/jit.html#inspecting-code
    with PathManager.open(os.path.join(dir, "model_ts_code.txt"), "w") as f:

        def get_code(mod):
            # Try a few ways to get code using private attributes.
            try:
                # This contains more information than just `mod.code`
                return _get_script_mod(mod)._c.code
            except AttributeError:
                pass
            try:
                return mod.code
            except AttributeError:
                return None

        def dump_code(prefix, mod):
            code = get_code(mod)
            name = prefix or "root model"
            if code is None:
                f.write(f"Could not found code for {name} (type={mod.original_name})\n")
                f.write("\n")
            else:
                f.write(f"\nCode for {name}, type={mod.original_name}:\n")
                f.write(code)
                f.write("\n")
                f.write("-" * 80)

            for name, m in mod.named_children():
                dump_code(prefix + "." + name, m)

        if isinstance(model, torch.jit.ScriptFunction):
            f.write(get_code(model))
        else:
            dump_code("", model)

    def _get_graph(model):
        try:
            # Recursively dump IR of all modules
            return _get_script_mod(model)._c.dump_to_str(True, False, False)
        except AttributeError:
            return model.graph.str()

    with PathManager.open(os.path.join(dir, "model_ts_IR.txt"), "w") as f:
        f.write(_get_graph(model))

    # Dump IR of the entire graph (all submodules inlined)
    with PathManager.open(os.path.join(dir, "model_ts_IR_inlined.txt"), "w") as f:
        f.write(str(model.inlined_graph))

    if not isinstance(model, torch.jit.ScriptFunction):
        # Dump the model structure in pytorch style
        with PathManager.open(os.path.join(dir, "model.txt"), "w") as f:
            f.write(str(model))
