import tqdm

from modules import shared


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=shared.state.job_count * shared.state.sampling_steps,
            position=1,
            file=shared.progress_print_out
        )

    def update(self):
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total = new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.refresh()
            self._tqdm.close()
            self._tqdm = None

