"""

Contains the functions that run when `gradio` is called from the command line. Specifically, allows

$ gradio app.py, to run app.py in reload mode where any changes in the app.py file or Gradio library reloads the demo.
$ gradio app.py my_demo.app, to use variable names other than "demo"
"""
import inspect
import os
import sys
from pathlib import Path

from uvicorn import Config
from uvicorn.supervisors import ChangeReload

import gradio
from gradio import networking, utils


def _setup_config():
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("No file specified.")
    if len(args) == 1 or args[1].startswith("--"):
        demo_name = "demo.app"
    else:
        demo_name = args[1]
        if "." not in demo_name:
            print(
                "\nWARNING: As of Gradio 3.31, the parameter after the file path must be the name of the FastAPI app, not the Gradio demo. In most cases, this just means you should add '.app' after the name of your demo, e.g. 'demo' -> 'demo.app'."
            )

    original_path = args[0]
    abs_original_path = utils.abspath(original_path)
    path = os.path.normpath(original_path)
    path = path.replace("/", ".")
    path = path.replace("\\", ".")
    filename = os.path.splitext(path)[0]

    gradio_folder = Path(inspect.getfile(gradio)).parent

    port = networking.get_first_available_port(
        networking.INITIAL_PORT_VALUE,
        networking.INITIAL_PORT_VALUE + networking.TRY_NUM_PORTS,
    )
    print(
        f"\nLaunching in *reload mode* on: http://{networking.LOCALHOST_NAME}:{port} (Press CTRL+C to quit)\n"
    )

    gradio_app = f"{filename}:{demo_name}"
    message = "Watching:"
    message_change_count = 0

    watching_dirs = []
    if str(gradio_folder).strip():
        watching_dirs.append(gradio_folder)
        message += f" '{gradio_folder}'"
        message_change_count += 1

    abs_parent = abs_original_path.parent
    if str(abs_parent).strip():
        watching_dirs.append(abs_parent)
        if message_change_count == 1:
            message += ","
        message += f" '{abs_parent}'"

    print(message + "\n")

    # guaranty access to the module of an app
    sys.path.insert(0, os.getcwd())

    # uvicorn.run blocks the execution (looping) which makes it hard to test
    return Config(
        gradio_app,
        reload=True,
        port=port,
        log_level="warning",
        reload_dirs=watching_dirs,
    )


def main():
    # default execution pattern to start the server and watch changes
    config = _setup_config()
    server = networking.Server(config)
    sock = config.bind_socket()
    ChangeReload(config, target=server.run, sockets=[sock]).run()


if __name__ == "__main__":
    main()
