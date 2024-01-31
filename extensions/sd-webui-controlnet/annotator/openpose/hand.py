import cv2
import json
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage.measure import label

from .model import handpose_model
from . import util

class Hand(object):
    def __init__(self, model_path):
        self.model = handpose_model()
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
            # print('cuda')
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImgRaw):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize for x in scale_search]

        wsize = 128
        heatmap_avg = np.zeros((wsize, wsize, 22))

        Hr, Wr, Cr = oriImgRaw.shape

        oriImg = cv2.GaussianBlur(oriImgRaw, (0, 0), 0.8)

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = util.smart_resize(oriImg, (scale, scale))

            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()

            with torch.no_grad():
                data = data.to(self.cn_device)
                output = self.model(data).cpu().numpy()

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))  # output 1 is heatmaps
            heatmap = util.smart_resize_k(heatmap, fx=stride, fy=stride)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = util.smart_resize(heatmap, (wsize, wsize))

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)

            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            y = int(float(y) * float(Hr) / float(wsize))
            x = int(float(x) * float(Wr) / float(wsize))
            all_peaks.append([x, y])
        return np.array(all_peaks)

if __name__ == "__main__":
    hand_estimation = Hand('../model/hand_pose_model.pth')

    # test_image = '../images/hand.jpg'
    test_image = '../images/hand.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    peaks = hand_estimation(oriImg)
    canvas = util.draw_handpose(oriImg, peaks, True)
    cv2.imshow('', canvas)
    cv2.waitKey(0)