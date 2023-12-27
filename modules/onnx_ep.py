from enum import Enum
from typing import Tuple, List
import onnxruntime as ort
from installer import log


class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    DirectML = "DmlExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCm = "ROCMExecutionProvider"
    OpenVINO = "OpenVINOExecutionProvider"


available_execution_providers: List[ExecutionProvider] = ort.get_available_providers()
EP_TO_NAME = {
    ExecutionProvider.CPU: "cpu", # is this a valid option?
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-cuda", # test required
    ExecutionProvider.ROCm: "gpu-rocm", # test required
    ExecutionProvider.OpenVINO: "gpu-openvino??", # test required
}


def get_default_execution_provider() -> ExecutionProvider:
    from modules import devices

    if devices.backend == "cpu":
        return ExecutionProvider.CPU
    elif devices.backend == "directml":
        return ExecutionProvider.DirectML
    elif devices.backend == "cuda":
        return ExecutionProvider.CUDA
    elif devices.backend == "rocm":
        if ExecutionProvider.ROCm in available_execution_providers:
            return ExecutionProvider.ROCm
        else:
            log.warning("Currently, there's no pypi release for onnxruntime-rocm. Please download and install .whl file from https://download.onnxruntime.ai/")
    elif devices.backend == "ipex" or devices.backend == "openvino":
        return ExecutionProvider.OpenVINO
    return ExecutionProvider.CPU


def get_execution_provider_options():
    from modules.shared import cmd_opts, opts

    execution_provider_options = {
        "device_id": int(cmd_opts.device_id or 0),
    }

    if opts.onnx_execution_provider == ExecutionProvider.ROCm:
        if ExecutionProvider.ROCm in available_execution_providers:
            execution_provider_options["tunable_op_enable"] = 1
            execution_provider_options["tunable_op_tuning_enable"] = 1
        else:
            log.warning("Currently, there's no pypi release for onnxruntime-rocm. Please download and install .whl file from https://download.onnxruntime.ai/ The inference will be fall back to CPU.")
    elif opts.onnx_execution_provider == ExecutionProvider.OpenVINO:
        from modules.intel.openvino import get_device as get_raw_openvino_device
        raw_openvino_device = get_raw_openvino_device()
        if opts.olive_float16 and not opts.openvino_hetero_gpu:
            raw_openvino_device = f"{raw_openvino_device}_FP16"
        execution_provider_options["device_type"] = raw_openvino_device
        del execution_provider_options["device_id"]

    return execution_provider_options


def get_provider() -> Tuple:
    from modules.shared import opts

    return (opts.onnx_execution_provider, get_execution_provider_options(),)
