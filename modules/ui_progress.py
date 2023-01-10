import time

import gradio as gr

from modules.shared import opts

import modules.shared as shared


def calc_time_left(progress, threshold, label, force_display, show_eta):
    if progress == 0:
        return ""
    else:
        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start
        if (eta_relative > threshold and show_eta) or force_display:
            if eta_relative > 3600:
                return label + time.strftime('%H:%M:%S', time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime('%M:%S',  time.gmtime(eta_relative))
            else:
                return label + time.strftime('%Ss',  time.gmtime(eta_relative))
        else:
            return ""


def check_progress_call(id_part):
    if shared.state.job_count == 0:
        return "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    # Show progress percentage and time left at the same moment, and base it also on steps done
    show_eta = progress >= 0.01 or shared.state.sampling_step >= 10

    time_left = calc_time_left(progress, 1, " ETA: ", shared.state.time_left_force_display, show_eta)
    if time_left != "":
        shared.state.time_left_force_display = True

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress*100))+"%" + time_left if show_eta else ""}</div></div>"""

    image = gr.update(visible=False)
    preview_visibility = gr.update(visible=False)

    if opts.show_progress_every_n_steps != 0:
        shared.state.set_current_image()
        image = shared.state.current_image

        if image is None:
            image = gr.update(value=None)
        else:
            preview_visibility = gr.update(visible=True)

    if shared.state.textinfo is not None:
        textinfo_result = gr.HTML.update(value=shared.state.textinfo, visible=True)
    else:
        textinfo_result = gr.update(visible=False)

    return f"<span id='{id_part}_progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image, textinfo_result


def check_progress_call_initial(id_part):
    shared.state.job_count = -1
    shared.state.current_latent = None
    shared.state.current_image = None
    shared.state.textinfo = None
    shared.state.time_start = time.time()
    shared.state.time_left_force_display = False

    return check_progress_call(id_part)


def setup_progressbar(progressbar, preview, id_part, textinfo=None):
    if textinfo is None:
        textinfo = gr.HTML(visible=False)

    check_progress = gr.Button('Check progress', elem_id=f"{id_part}_check_progress", visible=False)
    check_progress.click(
        fn=lambda: check_progress_call(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )

    check_progress_initial = gr.Button('Check progress (first)', elem_id=f"{id_part}_check_progress_initial", visible=False)
    check_progress_initial.click(
        fn=lambda: check_progress_call_initial(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )
