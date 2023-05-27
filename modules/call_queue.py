import html
import threading
import time
import cProfile
import pstats
import io
from rich import print # pylint: disable=redefined-builtin
from modules import shared, progress, errors

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)
        return res
    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):
        # if the first argument is a string that says "task(...)", it is treated as a job id
        if len(args) > 0 and type(args[0]) == str and args[0][0:5] == "task(" and args[0][-1] == ")":
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None
        with queue_lock:
            shared.state.begin()
            progress.start_task(id_task)
            res = [None, '', '', '']
            try:
                res = func(*args, **kwargs)
                progress.record_results(id_task, res)
            except Exception as e:
                shared.log.error(f"Exception: {e}")
                shared.log.error(f"Arguments: args={str(args)[:10240]} kwargs={str(kwargs)[:10240]}")
                errors.display(e, 'gradio call')
                res[-1] = f"<div class='error'>{html.escape(str(e))}</div>"
            finally:
                progress.finish_task(id_task)
            shared.state.end()
        return res
    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()
        try:
            if shared.cmd_opts.profile:
                pr = cProfile.Profile()
                pr.enable()
            res = func(*args, **kwargs)
            if res is None:
                msg = "No result returned from function"
                shared.log.warning(msg)
                res = [None, '', '', f"<div class='error'>{html.escape(msg)}</div>"]
            else:
                res = list(res)
            if shared.cmd_opts.profile:
                pr.disable()
                s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(15)
                print('Profile Exec:', s.getvalue())
        except Exception as e:
            errors.display(e, 'gradio call')
            shared.state.job = ""
            shared.state.job_count = 0
            if extra_outputs_array is None:
                extra_outputs_array = [None, '']
            res = extra_outputs_array + [f"<div class='error'>{html.escape(type(e).__name__+': '+str(e))}</div>"]
        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0
        if not add_stats:
            return tuple(res)
        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.2f}s"
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m}m "+elapsed_text
        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            vram_html = f" | <p class='vram'>GPU active {active_peak} MB reserved {reserved_peak} MB | System peak {sys_peak} MB total {sys_total} MB</p>"
        else:
            vram_html = ''
        res[-1] += f"<div class='performance'><p class='time'>Time taken: {elapsed_text}</p>{vram_html}</div>"
        return tuple(res)
    return f
