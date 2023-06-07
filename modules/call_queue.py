from functools import wraps
import html
import threading
import time

from modules import shared, progress, errors

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    @wraps(func)
    def f(*args, **kwargs):

        # if the first argument is a string that says "task(...)", it is treated as a job id
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None

        with queue_lock:
            shared.state.begin()
            progress.start_task(id_task)

            try:
                res = func(*args, **kwargs)
                progress.record_results(id_task, res)
            finally:
                progress.finish_task(id_task)

            shared.state.end()

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    @wraps(func)
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        if add_stats:
            shared.mem_mon.poll_rate = shared.opts.memmon_poll_rate
            shared.mem_mon.monitor()
        start_time = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            # When printing out our debug argument list,
            # do not print out more than a 100 KB of text
            max_debug_str_len = 131072
            message = "Error completing request"
            arg_str = f"Arguments: {args} {kwargs}"[:max_debug_str_len]
            if len(arg_str) > max_debug_str_len:
                arg_str += f" (Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)"
            errors.report(f"{message}\n{arg_str}", exc_info=True)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message = f'{type(e).__name__}: {e}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]

        mem_stats = shared.mem_mon.read()
        shared.mem_mon.stop()
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        if add_stats:
            # last item is always HTML
            from modules.performance_stats import format_performance_html
            res[-1] += format_performance_html(start_time=start_time, mem_stats=mem_stats)

        return tuple(res)

    return f
