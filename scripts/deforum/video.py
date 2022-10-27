import os
import pathlib

def vid2frames(video_path, frames_path, n=1, overwrite=True):      
    if not os.path.exists(frames_path) or overwrite: 
        try:
            for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
            	f.unlink()
        except:
            pass
        assert os.path.exists(video_path), f"Video input {video_path} does not exist"
          
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        t=1
        success = True
        while success:
            if count % n == 0:
                cv2.imwrite(frames_path + os.path.sep + f"{t:05}.jpg" , image)     # save frame as JPEG file
                t += 1
        success,image = vidcap.read()
        count += 1
        print("Converted %d frames" % count)
    else: print("Frames already unpacked")