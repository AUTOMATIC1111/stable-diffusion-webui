import sys
import traceback


def print_error_explanation(message):
    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])

    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)


def display(e: Exception, task):
    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

    message = str(e)
    if "copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])" in message:
        print_error_explanation("""
The most likely cause of this is you are trying to load Stable Diffusion 2.0 model without specifying its connfig file.
See https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20 for how to solve this.
        """)


def run(code, task):
    try:
        code()
    except Exception as e:
        display(task, e)
