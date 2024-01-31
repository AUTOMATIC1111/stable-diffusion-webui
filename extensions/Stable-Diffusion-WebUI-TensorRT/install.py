import launch
import sys
from importlib_metadata import version

python = sys.executable


def install():
    if launch.is_installed("tensorrt"):
        if not version("tensorrt") == "9.0.1.post11.dev4":
            launch.run(
                ["python", "-m", "pip", "uninstall", "-y", "tensorrt"],
                "removing old version of tensorrt",
            )

    if not launch.is_installed("tensorrt"):
        print("TensorRT is not installed! Installing...")
        launch.run_pip(
            "install nvidia-cudnn-cu11==8.9.4.25 --no-cache-dir", "nvidia-cudnn-cu11"
        )
        launch.run_pip(
            "install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir",
            "tensorrt",
            live=True,
        )
        launch.run(
            ["python", "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11"],
            "removing nvidia-cudnn-cu11",
        )

    if launch.is_installed("nvidia-cudnn-cu11"):
        if version("nvidia-cudnn-cu11") == "8.9.4.25":
            launch.run(
                ["python", "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11"],
                "removing nvidia-cudnn-cu11",
            )

    # Polygraphy
    if not launch.is_installed("polygraphy"):
        print("Polygraphy is not installed! Installing...")
        launch.run_pip(
            "install polygraphy --extra-index-url https://pypi.ngc.nvidia.com",
            "polygraphy",
            live=True,
        )

    # ONNX GS
    if not launch.is_installed("onnx_graphsurgeon"):
        print("GS is not installed! Installing...")
        launch.run_pip("install protobuf==3.20.2", "protobuf", live=True)
        launch.run_pip(
            "install onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com",
            "onnx-graphsurgeon",
            live=True,
        )

    # OPTIMUM
    if not launch.is_installed("optimum"):
        print("Optimum is not installed! Installing...")
        launch.run_pip(
            "install optimum",
            "optimum",
            live=True,
        )


install()
