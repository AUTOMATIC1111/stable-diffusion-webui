# Author: Filarius and orcist1
# adjusted from https://github.com/Filarius

# import math
# from fileinput import filename
import os
import sys

# import traceback
import random

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image

# from modules.shared import opts, cmd_opts, state
from modules.shared import state
from modules import processing

from subprocess import Popen, PIPE
import numpy as np


class Script(scripts.Script):
    def title(self):
        return "[C] Video to video"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_path = gr.Textbox(label="Input file path", lines=1)
        # output_path = gr.Textbox(label="Output file path", lines=1)
        crf = gr.Slider(
            label="CRF (quality, less is better, x264 param)",
            minimum=1,
            maximum=40,
            step=1,
            value=24,
        )
        fps = gr.Slider(
            label="FPS",
            minimum=1,
            maximum=60,
            step=1,
            value=24,
        )

        with gr.Row():
            seed_walk = gr.Slider(
                minimum=0, maximum=20, step=1, label="Seed step size", value=3
            )
            seed_max_distance = gr.Slider(
                minimum=3, maximum=100, step=1, label="Seed max distance", value=15
            )

        with gr.Row():
            start_time = gr.Textbox(label="Start time", value="00:00:00", lines=1)
            end_time = gr.Textbox(label="End time", value="00:00:00", lines=1)

        return [
            input_path,
            crf,
            fps,
            seed_walk,
            seed_max_distance,
            start_time,
            end_time,
        ]

    def run(
        self,
        p,
        input_path,
        crf,
        fps,
        seed_walk,
        seed_max_distance,
        start_time,
        end_time,
    ):
        processing.fix_seed(p)

        # p.subseed_strength == 0
        initial_seed = p.seed
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.batch_count = 1

        start_time = start_time.strip()
        end_time = end_time.strip()

        if start_time == "":
            start_time = "00:00:00"
        if end_time == "00:00:00":
            end_time = ""

        time_interval = (
            f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else ""
        )

        import modules

        path = modules.paths.script_path
        save_dir = "outputs/img2img-video/"
        ffmpeg.install(path, save_dir)

        input_file = os.path.normpath(input_path.strip())
        decoder = ffmpeg(
            " ".join(
                [
                    "ffmpeg/ffmpeg -y -loglevel panic",
                    f'{time_interval} -i "{input_file}"',
                    f"-s:v {p.width}x{p.height} -r {fps}",
                    "-f image2pipe -pix_fmt rgb24",
                    "-vcodec rawvideo -",
                ]
            ),
            use_stdout=True,
        )
        decoder.start()

        output_file = input_file.split("\\")[-1]
        encoder = ffmpeg(
            " ".join(
                [
                    "ffmpeg/ffmpeg -y -loglevel panic",
                    "-f rawvideo -pix_fmt rgb24",
                    f"-s:v {p.width}x{p.height} -r {fps}",
                    "-i - -c:v libx264 -preset fast",
                    f'-crf {crf} "{save_dir}/{output_file}"',
                ]
            ),
            use_stdin=True,
        )
        encoder.start()

        batch = []
        seed = initial_seed

        frame = 1
        seconds = ffmpeg.seconds(end_time) - ffmpeg.seconds(start_time)
        loops = seconds * int(fps)
        state.job_count = loops

        pull_count = p.width * p.height * 3
        raw_image = decoder.readout(pull_count)

        while raw_image is not None and len(raw_image) > 0:
            image_PIL = Image.fromarray(
                np.uint8(raw_image).reshape((p.height, p.width, 3)), mode="RGB"
            )
            batch.append(image_PIL)

            if len(batch) == p.batch_size:
                state.job = f"{frame}/{loops}:"

                seed_step = (
                    random.randint(0, seed_walk) * 1 if random.randint(0, 1) else -1
                )
                if abs(seed + seed_step - initial_seed) <= seed_max_distance:
                    seed = seed + seed_step
                p.seed = [seed for _ in batch]

                p.init_images = batch
                batch = []
                proc = process_images(p)

                for output in proc.images:
                    encoder.write(np.asarray(output))

            raw_image = decoder.readout(pull_count)
            frame += 1

        return Processed(p, [], p.seed, "")


class ffmpeg:
    def __init__(
        self,
        cmdln,
        use_stdin=False,
        use_stdout=False,
        use_stderr=False,
        print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        self._process = Popen(
            self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr
        )

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None

    @staticmethod
    def install(path, save_dir):
        from basicsr.utils.download_util import load_file_from_url
        from zipfile import ZipFile

        ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/5.1.1/ffmpeg-5.1.1-full_build.zip"
        ffmpeg_dir = os.path.join(path, "ffmpeg")

        ckpt_path = load_file_from_url(url=ffmpeg_url, model_dir=ffmpeg_dir)

        if not os.path.exists(os.path.abspath(os.path.join(ffmpeg_dir, "ffmpeg.exe"))):
            with ZipFile(ckpt_path, "r") as zipObj:
                listOfFileNames = zipObj.namelist()
                for fileName in listOfFileNames:
                    if "/bin/" in fileName:
                        zipObj.extract(fileName, ffmpeg_dir)
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffmpeg.exe"),
                os.path.join(ffmpeg_dir, "ffmpeg.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffplay.exe"),
                os.path.join(ffmpeg_dir, "ffplay.exe"),
            )
            os.rename(
                os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin", "ffprobe.exe"),
                os.path.join(ffmpeg_dir, "ffprobe.exe"),
            )

            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1], "bin"))
            os.rmdir(os.path.join(ffmpeg_dir, listOfFileNames[0][:-1]))
        os.makedirs(save_dir, exist_ok=True)
        return

    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds
