import os
import torch
from openvino.frontend.pytorch.torchdynamo.execute import execute, partitioned_modules, compiled_cache
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx
from hashlib import sha256

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    try:
        executor_parameters = None
        core = Core()
        if os.getenv("OPENVINO_TORCH_MODEL_CACHING") != "0":
            os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "1")
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}

        example_inputs.reverse()
        cache_root = "./cache/"
        if os.getenv("OPENVINO_TORCH_CACHE_DIR") is not None:
            cache_root = os.getenv("OPENVINO_TORCH_CACHE_DIR")

        device = "GPU"

        if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None:
            device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
            assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"
        else:
            os.environ.setdefault('OPENVINO_TORCH_BACKEND_DEVICE', device)

        #Cache saving keeps increasing the partition id
        #This loop check if non 0 partition id caches exist
        #Takes 0.002 seconds when nothing is found
        use_cached_file = False
        for i in range(100):
            file_name = get_cached_file_name(*example_inputs, model_hash_str=str(model_hash_str + str(i)), device=device, cache_root=cache_root)
            if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
                use_cached_file = True
                break

        if use_cached_file:
            om = core.read_model(file_name + ".xml")

            dtype_mapping = {
                torch.float32: Type.f32,
                torch.float64: Type.f64,
                torch.float16: Type.f16,
                torch.int64: Type.i64,
                torch.int32: Type.i32,
                torch.uint8: Type.u8,
                torch.int8: Type.i8,
                torch.bool: Type.boolean
            }

            for idx, input_data in enumerate(example_inputs):
                om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
                om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
            om.validate_nodes_and_infer_types()

            if model_hash_str is not None:
                core.set_property({'CACHE_DIR': cache_root + '/blob'})

            compiled_model = core.compile_model(om, device)
            def _call(*args):
                ov_inputs = [a.detach().cpu().numpy() for a in args]
                ov_inputs.reverse()
                res = compiled_model(ov_inputs)
                result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
                return result
            return _call
        else:
            example_inputs.reverse()
            model = make_fx(subgraph)(*example_inputs)
            with torch.no_grad():
                model.eval()
            partitioner = Partitioner()
            compiled_model = partitioner.make_partitions(model)

            def _call(*args):
                res = execute(compiled_model, *args, executor="openvino",
                              executor_parameters=executor_parameters)
                return res
            return _call
    except Exception:
        return compile_fx(subgraph, example_inputs)


def get_cached_file_name(*args, model_hash_str, device, cache_root):
    file_name = None
    if model_hash_str is not None:
        model_cache_dir = cache_root + "/model/"
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
            file_name = model_cache_dir + model_hash_str + "_" + device
            for input_data in args:
                if file_name is not None:
                    file_name += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
        except OSError as error:
            print("Cache directory ", cache_root, " cannot be created. Model caching is disabled. Error: ", error)
            file_name = None
            model_hash_str = None
    return file_name

def openvino_clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()
