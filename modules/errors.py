import logging
import warnings
from rich.console import Console
from rich.theme import Theme
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from installer import log as installer_log, setup_logging


setup_logging()
log = installer_log
console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
    "traceback.border": "black",
    "traceback.border.syntax_error": "black",
    "inspect.value.border": "black",
}))

pretty_install(console=console)
traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False)
already_displayed = {}


def install(suppress=[]): # noqa: B006
    warnings.filterwarnings("ignore", category=UserWarning)
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=suppress)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s')
    # for handler in logging.getLogger().handlers:
    #    handler.setLevel(logging.INFO)


def print_error_explanation(message):
    lines = message.strip().split("\n")
    for line in lines:
        log.error(line)


def display(e: Exception, task, suppress=[]): # noqa: B006
    log.error(f"{task or 'error'}: {type(e).__name__}")
    console.print_exception(show_locals=False, max_frames=10, extra_lines=1, suppress=suppress, theme="ansi_dark", word_wrap=False, width=min([console.width, 200]))


def display_once(e: Exception, task):
    if task in already_displayed:
        return
    display(e, task)
    already_displayed[task] = 1


def run(code, task):
    try:
        code()
    except Exception as e:
        display(e, task)


def exception(suppress=[]): # noqa: B006
    console.print_exception(show_locals=False, max_frames=10, extra_lines=2, suppress=suppress, theme="ansi_dark", word_wrap=False, width=min([console.width, 200]))


def profile(profiler, msg: str):
    profiler.disable()
    import io
    import pstats
    stream = io.StringIO() # pylint: disable=abstract-class-instantiated
    p = pstats.Stats(profiler, stream=stream)
    p.sort_stats(pstats.SortKey.CUMULATIVE)
    p.print_stats(100)
    # p.print_title()
    # p.print_call_heading(10, 'time')
    # p.print_callees(10)
    # p.print_callers(10)
    profiler = None
    lines = stream.getvalue().split('\n')
    lines = [x for x in lines if '<frozen' not in x
             and '{built-in' not in x
             and '/logging' not in x
             and 'Ordered by' not in x
             and 'List reduced' not in x
             and '_lsprof' not in x
             and '/profiler' not in x
             and 'rich' not in x
             and x.strip() != ''
            ]
    txt = '\n'.join(lines[:min(5, len(lines))])
    log.debug(f'Profile {msg}: {txt}')


def profile_torch(profiler, msg: str):
    profiler.stop()
    lines = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
    lines = lines.split('\n')
    lines = [x for x in lines if '/profiler' not in x and '---' not in x]
    txt = '\n'.join(lines)
    # print(f'Torch {msg}:', txt)
    log.debug(f'Torch profile {msg}: \n{txt}')
