# Copyright (c) Facebook, Inc. and its affiliates.
import collections
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import torch
from torch import nn

from annotator.oneformer.detectron2.structures import Boxes, Instances, ROIMasks
from annotator.oneformer.detectron2.utils.registry import _convert_target_to_string, locate

from .torchscript_patch import patch_builtin_len


@dataclass
class Schema:
    """
    A Schema defines how to flatten a possibly hierarchical object into tuple of
    primitive objects, so it can be used as inputs/outputs of PyTorch's tracing.

    PyTorch does not support tracing a function that produces rich output
    structures (e.g. dict, Instances, Boxes). To trace such a function, we
    flatten the rich object into tuple of tensors, and return this tuple of tensors
    instead. Meanwhile, we also need to know how to "rebuild" the original object
    from the flattened results, so we can evaluate the flattened results.
    A Schema defines how to flatten an object, and while flattening it, it records
    necessary schemas so that the object can be rebuilt using the flattened outputs.

    The flattened object and the schema object is returned by ``.flatten`` classmethod.
    Then the original object can be rebuilt with the ``__call__`` method of schema.

    A Schema is a dataclass that can be serialized easily.
    """

    # inspired by FetchMapper in tensorflow/python/client/session.py

    @classmethod
    def flatten(cls, obj):
        raise NotImplementedError

    def __call__(self, values):
        raise NotImplementedError

    @staticmethod
    def _concat(values):
        ret = ()
        sizes = []
        for v in values:
            assert isinstance(v, tuple), "Flattened results must be a tuple"
            ret = ret + v
            sizes.append(len(v))
        return ret, sizes

    @staticmethod
    def _split(values, sizes):
        if len(sizes):
            expected_len = sum(sizes)
            assert (
                len(values) == expected_len
            ), f"Values has length {len(values)} but expect length {expected_len}."
        ret = []
        for k in range(len(sizes)):
            begin, end = sum(sizes[:k]), sum(sizes[: k + 1])
            ret.append(values[begin:end])
        return ret


@dataclass
class ListSchema(Schema):
    schemas: List[Schema]  # the schemas that define how to flatten each element in the list
    sizes: List[int]  # the flattened length of each element

    def __call__(self, values):
        values = self._split(values, self.sizes)
        if len(values) != len(self.schemas):
            raise ValueError(
                f"Values has length {len(values)} but schemas " f"has length {len(self.schemas)}!"
            )
        values = [m(v) for m, v in zip(self.schemas, values)]
        return list(values)

    @classmethod
    def flatten(cls, obj):
        res = [flatten_to_tuple(k) for k in obj]
        values, sizes = cls._concat([k[0] for k in res])
        return values, cls([k[1] for k in res], sizes)


@dataclass
class TupleSchema(ListSchema):
    def __call__(self, values):
        return tuple(super().__call__(values))


@dataclass
class IdentitySchema(Schema):
    def __call__(self, values):
        return values[0]

    @classmethod
    def flatten(cls, obj):
        return (obj,), cls()


@dataclass
class DictSchema(ListSchema):
    keys: List[str]

    def __call__(self, values):
        values = super().__call__(values)
        return dict(zip(self.keys, values))

    @classmethod
    def flatten(cls, obj):
        for k in obj.keys():
            if not isinstance(k, str):
                raise KeyError("Only support flattening dictionaries if keys are str.")
        keys = sorted(obj.keys())
        values = [obj[k] for k in keys]
        ret, schema = ListSchema.flatten(values)
        return ret, cls(schema.schemas, schema.sizes, keys)


@dataclass
class InstancesSchema(DictSchema):
    def __call__(self, values):
        image_size, fields = values[-1], values[:-1]
        fields = super().__call__(fields)
        return Instances(image_size, **fields)

    @classmethod
    def flatten(cls, obj):
        ret, schema = super().flatten(obj.get_fields())
        size = obj.image_size
        if not isinstance(size, torch.Tensor):
            size = torch.tensor(size)
        return ret + (size,), schema


@dataclass
class TensorWrapSchema(Schema):
    """
    For classes that are simple wrapper of tensors, e.g.
    Boxes, RotatedBoxes, BitMasks
    """

    class_name: str

    def __call__(self, values):
        return locate(self.class_name)(values[0])

    @classmethod
    def flatten(cls, obj):
        return (obj.tensor,), cls(_convert_target_to_string(type(obj)))


# if more custom structures needed in the future, can allow
# passing in extra schemas for custom types
def flatten_to_tuple(obj):
    """
    Flatten an object so it can be used for PyTorch tracing.
    Also returns how to rebuild the original object from the flattened outputs.

    Returns:
        res (tuple): the flattened results that can be used as tracing outputs
        schema: an object with a ``__call__`` method such that ``schema(res) == obj``.
             It is a pure dataclass that can be serialized.
    """
    schemas = [
        ((str, bytes), IdentitySchema),
        (list, ListSchema),
        (tuple, TupleSchema),
        (collections.abc.Mapping, DictSchema),
        (Instances, InstancesSchema),
        ((Boxes, ROIMasks), TensorWrapSchema),
    ]
    for klass, schema in schemas:
        if isinstance(obj, klass):
            F = schema
            break
    else:
        F = IdentitySchema

    return F.flatten(obj)


class TracingAdapter(nn.Module):
    """
    A model may take rich input/output format (e.g. dict or custom classes),
    but `torch.jit.trace` requires tuple of tensors as input/output.
    This adapter flattens input/output format of a model so it becomes traceable.

    It also records the necessary schema to rebuild model's inputs/outputs from flattened
    inputs/outputs.

    Example:
    ::
        outputs = model(inputs)   # inputs/outputs may be rich structure
        adapter = TracingAdapter(model, inputs)

        # can now trace the model, with adapter.flattened_inputs, or another
        # tuple of tensors with the same length and meaning
        traced = torch.jit.trace(adapter, adapter.flattened_inputs)

        # traced model can only produce flattened outputs (tuple of tensors)
        flattened_outputs = traced(*adapter.flattened_inputs)
        # adapter knows the schema to convert it back (new_outputs == outputs)
        new_outputs = adapter.outputs_schema(flattened_outputs)
    """

    flattened_inputs: Tuple[torch.Tensor] = None
    """
    Flattened version of inputs given to this class's constructor.
    """

    inputs_schema: Schema = None
    """
    Schema of the inputs given to this class's constructor.
    """

    outputs_schema: Schema = None
    """
    Schema of the output produced by calling the given model with inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        inputs,
        inference_func: Optional[Callable] = None,
        allow_non_tensor: bool = False,
    ):
        """
        Args:
            model: an nn.Module
            inputs: An input argument or a tuple of input arguments used to call model.
                After flattening, it has to only consist of tensors.
            inference_func: a callable that takes (model, *inputs), calls the
                model with inputs, and return outputs. By default it
                is ``lambda model, *inputs: model(*inputs)``. Can be override
                if you need to call the model differently.
            allow_non_tensor: allow inputs/outputs to contain non-tensor objects.
                This option will filter out non-tensor objects to make the
                model traceable, but ``inputs_schema``/``outputs_schema`` cannot be
                used anymore because inputs/outputs cannot be rebuilt from pure tensors.
                This is useful when you're only interested in the single trace of
                execution (e.g. for flop count), but not interested in
                generalizing the traced graph to new inputs.
        """
        super().__init__()
        if isinstance(model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        self.model = model
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        self.inputs = inputs
        self.allow_non_tensor = allow_non_tensor

        if inference_func is None:
            inference_func = lambda model, *inputs: model(*inputs)  # noqa
        self.inference_func = inference_func

        self.flattened_inputs, self.inputs_schema = flatten_to_tuple(inputs)

        if all(isinstance(x, torch.Tensor) for x in self.flattened_inputs):
            return
        if self.allow_non_tensor:
            self.flattened_inputs = tuple(
                [x for x in self.flattened_inputs if isinstance(x, torch.Tensor)]
            )
            self.inputs_schema = None
        else:
            for input in self.flattened_inputs:
                if not isinstance(input, torch.Tensor):
                    raise ValueError(
                        "Inputs for tracing must only contain tensors. "
                        f"Got a {type(input)} instead."
                    )

    def forward(self, *args: torch.Tensor):
        with torch.no_grad(), patch_builtin_len():
            if self.inputs_schema is not None:
                inputs_orig_format = self.inputs_schema(args)
            else:
                if len(args) != len(self.flattened_inputs) or any(
                    x is not y for x, y in zip(args, self.flattened_inputs)
                ):
                    raise ValueError(
                        "TracingAdapter does not contain valid inputs_schema."
                        " So it cannot generalize to other inputs and must be"
                        " traced with `.flattened_inputs`."
                    )
                inputs_orig_format = self.inputs

            outputs = self.inference_func(self.model, *inputs_orig_format)
            flattened_outputs, schema = flatten_to_tuple(outputs)

            flattened_output_tensors = tuple(
                [x for x in flattened_outputs if isinstance(x, torch.Tensor)]
            )
            if len(flattened_output_tensors) < len(flattened_outputs):
                if self.allow_non_tensor:
                    flattened_outputs = flattened_output_tensors
                    self.outputs_schema = None
                else:
                    raise ValueError(
                        "Model cannot be traced because some model outputs "
                        "cannot flatten to tensors."
                    )
            else:  # schema is valid
                if self.outputs_schema is None:
                    self.outputs_schema = schema
                else:
                    assert self.outputs_schema == schema, (
                        "Model should always return outputs with the same "
                        "structure so it can be traced!"
                    )
            return flattened_outputs

    def _create_wrapper(self, traced_model):
        """
        Return a function that has an input/output interface the same as the
        original model, but it calls the given traced model under the hood.
        """

        def forward(*args):
            flattened_inputs, _ = flatten_to_tuple(args)
            flattened_outputs = traced_model(*flattened_inputs)
            return self.outputs_schema(flattened_outputs)

        return forward
