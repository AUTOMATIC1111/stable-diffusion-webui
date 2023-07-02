import sys
from fastapi import FastAPI

import gradio as gr
from scripts.taskListener import TaskListener


def start_queue(_: gr.Blocks, app: FastAPI):
    if '--start-task-listener' in sys.argv:
        print(f"Launching API server with task listener")
        task = TaskListener()
        task.start()
    if '--start-prod-task-listener' in sys.argv:
        print(f"Launching API server with task listener")
        task_p = TaskListener("prod")
        task_p.start()


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(start_queue)
except:
    pass
