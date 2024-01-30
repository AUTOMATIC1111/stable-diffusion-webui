import sys
from enum import Enum
from typing import Tuple, List
import onnxruntime as ort
from installer import log


class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    DirectML = "DmlExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCm = "ROCMExecutionProvider"
    MIGraphX = "MIGraphXExecutionProvider"
    OpenVINO = "OpenVINOExecutionProvider"


available_execution_providers: List[ExecutionProvider] = ort.get_available_providers()
EP_TO_NAME = {
    ExecutionProvider.CPU: "gpu-cpu", # ???
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-cuda", # test required
    ExecutionProvider.ROCm: "gpu-rocm", # test required
    ExecutionProvider.MIGraphX: "gpu-migraphx", # test required
    ExecutionProvider.OpenVINO: "gpu-openvino??", # test required
}
TORCH_DEVICE_TO_EP = {
    "cpu": ExecutionProvider.CPU,
    "cuda": ExecutionProvider.CUDA,
    "privateuseone": ExecutionProvider.DirectML,
    "meta": None,
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
            return ExecutionProvider.CPU
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


def install_execution_provider(ep: ExecutionProvider):
    from installer import pip, uninstall, installed, get_onnxruntime_source_for_rocm

    if installed("onnxruntime"):
        uninstall("onnxruntime")
    if installed("onnxruntime-directml"):
        uninstall("onnxruntime-directml")
    if installed("onnxruntime-gpu"):
        uninstall("onnxruntime-gpu")
    if installed("onnxruntime-training"):
        uninstall("onnxruntime-training")
    if installed("onnxruntime-openvino"):
        uninstall("onnxruntime-openvino")

    packages = ["onnxruntime"] # Failed to load olive: cannot import name '__version__' from 'onnxruntime'

    if ep == ExecutionProvider.DirectML:
        packages.append("onnxruntime-directml")
    elif ep == ExecutionProvider.CUDA:
        packages.append("onnxruntime-gpu")
    elif ep == ExecutionProvider.ROCm:
        if "linux" not in sys.platform:
            log.warning("ROCMExecutionProvider is not supported on Windows.")
            return

        packages.append(get_onnxruntime_source_for_rocm(None))
    elif ep == ExecutionProvider.OpenVINO:
        if installed("openvino"):
            uninstall("openvino")
        packages.append("openvino")
        packages.append("onnxruntime-openvino")

    pip(f"install --upgrade {' '.join(packages)}")
    log.info("Please restart SD.Next.")
