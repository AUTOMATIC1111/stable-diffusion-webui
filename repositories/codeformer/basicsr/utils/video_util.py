'''
The code is modified from the Real-ESRGAN:
https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py

'''
import cv2
import sys
import numpy as np

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret

class VideoReader:
    def __init__(self, video_path):
        self.paths = []  # for image&folder type
        self.audio = None
        try:
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        except FileNotFoundError:
            print('Please install ffmpeg (not ffmpeg-python) by running\n',
                  '\t$ conda install -c conda-forge ffmpeg')
            sys.exit(0)

        meta = get_video_meta_info(video_path)
        self.width = meta['width']
        self.height = meta['height']
        self.input_fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']

        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        return self.get_frame_from_stream()


    def close(self):
        self.stream_reader.stdin.close()
        self.stream_reader.wait()


class VideoWriter:
    def __init__(self, video_save_path, height, width, fps, audio):
        if height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')
        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                            framerate=fps).output(
                                audio,
                                video_save_path,
                                pix_fmt='yuv420p',
                                vcodec='libx264',
                                loglevel='error',
                                acodec='copy').overwrite_output().run_async(
                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                            framerate=fps).output(
                                video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                loglevel='error').overwrite_output().run_async(
                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))

    def write_frame(self, frame):
        try:
            frame = frame.astype(np.uint8).tobytes()
            self.stream_writer.stdin.write(frame)
        except BrokenPipeError:
            print('Please re-install ffmpeg and libx264 by running\n',
                  '\t$ conda install -c conda-forge ffmpeg\n',
                  '\t$ conda install -c conda-forge x264')
            sys.exit(0)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()