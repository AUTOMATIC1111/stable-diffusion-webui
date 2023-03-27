import sys
import traceback
from rich.console import Console

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
    console = Console()
    console.print_exception(show_locals=False, max_frames=2, extra_lines=1, suppress=[], word_wrap=False, width=min([console.width, 200]))


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
