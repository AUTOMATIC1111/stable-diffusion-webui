from functools import wraps
import html
import time

from modules import shared, progress, errors, devices, fifo_lock

queue_lock = fifo_lock.FIFOLock()


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
            shared.state.begin(job=id_task)
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
        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

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

        devices.torch_gc()

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        if not add_stats:
            return tuple(res)

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = sys_peak/max(sys_total, 1) * 100

            toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
            toltip_r = "Reserved: total amout of video memory allocated by the Torch library "
            toltip_sys = "System: peak amout of video memory allocated by all running programs, out of total capacity"

            text_a = f"<abbr title='{toltip_a}'>A</abbr>: <span class='measurement'>{active_peak/1024:.2f} GB</span>"
            text_r = f"<abbr title='{toltip_r}'>R</abbr>: <span class='measurement'>{reserved_peak/1024:.2f} GB</span>"
            text_sys = f"<abbr title='{toltip_sys}'>Sys</abbr>: <span class='measurement'>{sys_peak/1024:.1f}/{sys_total/1024:g} GB</span> ({sys_pct:.1f}%)"

            vram_html = f"<p class='vram'>{text_a}, <wbr>{text_r}, <wbr>{text_sys}</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}</div>"

        return tuple(res)

    return f
