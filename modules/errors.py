import sys
import anyio
import starlette
import gradio
from rich import print
from rich.console import Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
pretty_install(console=console)
traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[anyio, starlette, gradio])
already_displayed = {}


def print_error_explanation(message):
    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])
    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)


def display(e: Exception, task):
    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    console.print_exception(show_locals=False, max_frames=2, extra_lines=1, suppress=[anyio, starlette, gradio], word_wrap=False, width=min([console.width, 200]))


def display_once(e: Exception, task):
    if task in already_displayed:
        return
    display(e, task)
    already_displayed[task] = 1


def run(code, task):
    try:
        code()
    except Exception as e:
        display(task, e)


def exception():
    console.print_exception(show_locals=False, max_frames=10, extra_lines=2, suppress=[anyio, starlette, gradio], word_wrap=False, width=min([console.width, 200]))
