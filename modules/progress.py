import base64
import io
import time

import gradio as gr
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import Depends, Security
from fastapi.security import APIKeyCookie

from modules import call_queue
from modules.shared import opts

import modules.shared as shared


current_task_user = None
current_task = None
pending_tasks = {}
finished_tasks = []


def start_task(user, id_task):
    global current_task
    global current_task_user

    current_task_user = user
    current_task = id_task
    pending_tasks.pop((user, id_task), None)


def finish_task(user, id_task):
    global current_task
    global current_task_user

    if current_task == id_task:
        current_task = None

    if current_task_user == user:
        current_task_user = None

    finished_tasks.append((user, id_task))
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)


def add_task_to_queue(user, id_job):
    pending_tasks[(user, id_job)] = time.time()

last_task_id = None
last_task_result = None
last_task_user = None

def set_last_task_result(user, id_job, result):

  global last_task_id
  global last_task_result
  global last_task_user

  last_task_id = id_job
  last_task_result = result
  last_task_user = user


def restore_progress_call(request: gr.Request):
    if current_task is None:

      # image, generation_info, html_info, html_log
      return tuple(list([None, None, None, None]))

    else:
      user = request.username

      if current_task_user == user:
        t_task = current_task
        with call_queue.queue_lock_condition:
          call_queue.queue_lock_condition.wait_for(lambda: t_task == last_task_id)

        return last_task_result

      return tuple(list([None, None, None, None]))

class CurrentTaskResponse(BaseModel):
  current_task: str = Field(default=None, title="Task ID", description="id of the current progress task")

class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")


class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float = Field(default=None, title="ETA in secs")
    live_preview: str = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)

def setup_current_task_api(app):

    def get_current_user(token: Optional[str] = Security(APIKeyCookie(name="access-token", auto_error=False))):
      return None if token is None else app.tokens.get(token)

    def current_task_api(current_user: str = Depends(get_current_user)):

      if app.auth is None or current_task_user == current_user:
        current_user_task = current_task
      else:
        current_user_task = None

      return CurrentTaskResponse(current_task=current_user_task)

    return app.add_api_route("/internal/current_task", current_task_api, methods=["GET"], response_model=CurrentTaskResponse)

def progressapi(req: ProgressRequest):
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks

    if not active:
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo="In queue..." if queued else "Waiting...")

    progress = 0

    job_count, job_no = shared.state.job_count, shared.state.job_no
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    elapsed_since_start = time.time() - shared.state.time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    id_live_preview = req.id_live_preview
    shared.state.set_current_image()
    if opts.live_previews_enable and shared.state.id_live_preview != req.id_live_preview:
        image = shared.state.current_image
        if image is not None:
            buffered = io.BytesIO()
            image.save(buffered, format="png")
            live_preview = 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode("ascii")
            id_live_preview = shared.state.id_live_preview
        else:
            live_preview = None
    else:
        live_preview = None

    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)