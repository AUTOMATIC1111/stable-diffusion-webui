import torch

from modules import shared, ui_gradio_extensions


class Profiler:
    def __init__(self):
        if not shared.opts.profiling_enable:
            self.profiler = None
            return

        activities = []
        if "CPU" in shared.opts.profiling_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "CUDA" in shared.opts.profiling_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        if not activities:
            self.profiler = None
            return

        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=shared.opts.profiling_record_shapes,
            profile_memory=shared.opts.profiling_profile_memory,
            with_stack=shared.opts.profiling_with_stack
        )

    def __enter__(self):
        if self.profiler:
            self.profiler.__enter__()

        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if self.profiler:
            shared.state.textinfo = "Finishing profile..."

            self.profiler.__exit__(exc_type, exc, exc_tb)

            self.profiler.export_chrome_trace(shared.opts.profiling_filename)


def webpath():
    return ui_gradio_extensions.webpath(shared.opts.profiling_filename)

