import sys

import gradio.deploy_space
import gradio.reload


def cli():
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("No file specified.")
    if args[0] == "deploy":
        gradio.deploy_space.deploy()
    else:
        gradio.reload.main()
