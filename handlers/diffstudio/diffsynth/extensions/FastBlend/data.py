import imageio, os
import numpy as np
from PIL import Image


def read_video(file_name):
    reader = imageio.get_reader(file_name)
    video = []
    for frame in reader:
        frame = np.array(frame)
        video.append(frame)
    reader.close()
    return video


def get_video_fps(file_name):
    reader = imageio.get_reader(file_name)
    fps = reader.get_meta_data()["fps"]
    reader.close()
    return fps


def save_video(frames_path, video_path, num_frames, fps):
    writer = imageio.get_writer(video_path, fps=fps, quality=9)
    for i in range(num_frames):
        frame = np.array(Image.open(os.path.join(frames_path, "%05d.png" % i)))
        writer.append_data(frame)
    writer.close()
    return video_path


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return np.array(self.reader.get_data(item))

    def __del__(self):
        self.reader.close()


def split_file_name(file_name):
    result = []
    number = -1
    for i in file_name:
        if ord(i)>=ord("0") and ord(i)<=ord("9"):
            if number == -1:
                number = 0
            number = number*10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result = tuple(result)
    return result


def search_for_images(folder):
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


def read_images(folder):
    file_list = search_for_images(folder)
    frames = [np.array(Image.open(i)) for i in file_list]
    return frames


class LowMemoryImageFolder:
    def __init__(self, folder, file_list=None):
        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        return np.array(Image.open(self.file_list[item]))

    def __del__(self):
        pass


class VideoData:
    def __init__(self, video_file, image_folder, **kwargs):
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.height = None
        self.width = None

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        frame = self.data.__getitem__(item)
        height, width, _ = frame.shape
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = Image.fromarray(frame).resize((self.width, self.height))
                frame = np.array(frame)
        return frame

    def __del__(self):
        pass
