import logging
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import torch
import torch.nn.functional as F
import cv2

from . import util
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, init


class FaceNet(Module):
    """Model the cascading heatmaps. """
    def __init__(self):
        super(FaceNet, self).__init__()
        # cnn to make feature map
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = Conv2d(in_channels=3, out_channels=64,
                              kernel_size=3, stride=1, padding=1)
        self.conv1_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1,
            padding=1)
        self.conv2_1 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv2_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv3_1 = Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1,
            padding=1)
        self.conv3_2 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1,
            padding=1)
        self.conv3_3 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1,
            padding=1)
        self.conv3_4 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1,
            padding=1)
        self.conv4_1 = Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1,
            padding=1)
        self.conv4_2 = Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1,
            padding=1)
        self.conv4_3 = Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1,
            padding=1)
        self.conv4_4 = Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1,
            padding=1)
        self.conv5_1 = Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1,
            padding=1)
        self.conv5_2 = Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1,
            padding=1)
        self.conv5_3_CPM = Conv2d(
            in_channels=512, out_channels=128, kernel_size=3, stride=1,
            padding=1)

        # stage1
        self.conv6_1_CPM = Conv2d(
            in_channels=128, out_channels=512, kernel_size=1, stride=1,
            padding=0)
        self.conv6_2_CPM = Conv2d(
            in_channels=512, out_channels=71, kernel_size=1, stride=1,
            padding=0)

        # stage2
        self.Mconv1_stage2 = Conv2d(
            in_channels=199, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv2_stage2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv3_stage2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv4_stage2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv5_stage2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv6_stage2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.Mconv7_stage2 = Conv2d(
            in_channels=128, out_channels=71, kernel_size=1, stride=1,
            padding=0)

        # stage3
        self.Mconv1_stage3 = Conv2d(
            in_channels=199, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv2_stage3 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv3_stage3 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv4_stage3 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv5_stage3 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv6_stage3 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.Mconv7_stage3 = Conv2d(
            in_channels=128, out_channels=71, kernel_size=1, stride=1,
            padding=0)

        # stage4
        self.Mconv1_stage4 = Conv2d(
            in_channels=199, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv2_stage4 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv3_stage4 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv4_stage4 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv5_stage4 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv6_stage4 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.Mconv7_stage4 = Conv2d(
            in_channels=128, out_channels=71, kernel_size=1, stride=1,
            padding=0)

        # stage5
        self.Mconv1_stage5 = Conv2d(
            in_channels=199, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv2_stage5 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv3_stage5 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv4_stage5 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv5_stage5 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv6_stage5 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.Mconv7_stage5 = Conv2d(
            in_channels=128, out_channels=71, kernel_size=1, stride=1,
            padding=0)

        # stage6
        self.Mconv1_stage6 = Conv2d(
            in_channels=199, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv2_stage6 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv3_stage6 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv4_stage6 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv5_stage6 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.Mconv6_stage6 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.Mconv7_stage6 = Conv2d(
            in_channels=128, out_channels=71, kernel_size=1, stride=1,
            padding=0)

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)

    def forward(self, x):
        """Return a list of heatmaps."""
        heatmaps = []

        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.relu(self.conv4_4(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3_CPM(h))
        feature_map = h

        # stage1
        h = self.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps.append(h)

        # stage2
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage2(h))
        h = self.relu(self.Mconv2_stage2(h))
        h = self.relu(self.Mconv3_stage2(h))
        h = self.relu(self.Mconv4_stage2(h))
        h = self.relu(self.Mconv5_stage2(h))
        h = self.relu(self.Mconv6_stage2(h))
        h = self.Mconv7_stage2(h)
        heatmaps.append(h)

        # stage3
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage3(h))
        h = self.relu(self.Mconv2_stage3(h))
        h = self.relu(self.Mconv3_stage3(h))
        h = self.relu(self.Mconv4_stage3(h))
        h = self.relu(self.Mconv5_stage3(h))
        h = self.relu(self.Mconv6_stage3(h))
        h = self.Mconv7_stage3(h)
        heatmaps.append(h)

        # stage4
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage4(h))
        h = self.relu(self.Mconv2_stage4(h))
        h = self.relu(self.Mconv3_stage4(h))
        h = self.relu(self.Mconv4_stage4(h))
        h = self.relu(self.Mconv5_stage4(h))
        h = self.relu(self.Mconv6_stage4(h))
        h = self.Mconv7_stage4(h)
        heatmaps.append(h)

        # stage5
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage5(h))
        h = self.relu(self.Mconv2_stage5(h))
        h = self.relu(self.Mconv3_stage5(h))
        h = self.relu(self.Mconv4_stage5(h))
        h = self.relu(self.Mconv5_stage5(h))
        h = self.relu(self.Mconv6_stage5(h))
        h = self.Mconv7_stage5(h)
        heatmaps.append(h)

        # stage6
        h = torch.cat([h, feature_map], dim=1)  # channel concat
        h = self.relu(self.Mconv1_stage6(h))
        h = self.relu(self.Mconv2_stage6(h))
        h = self.relu(self.Mconv3_stage6(h))
        h = self.relu(self.Mconv4_stage6(h))
        h = self.relu(self.Mconv5_stage6(h))
        h = self.relu(self.Mconv6_stage6(h))
        h = self.Mconv7_stage6(h)
        heatmaps.append(h)

        return heatmaps


LOG = logging.getLogger(__name__)
TOTEN = ToTensor()
TOPIL = ToPILImage()


params = {
    'gaussian_sigma': 2.5,
    'inference_img_size': 736,  # 368, 736, 1312
    'heatmap_peak_thresh': 0.1,
    'crop_scale': 1.5,
    'line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20], [20, 21],
        [22, 23], [23, 24], [24, 25], [25, 26],
        [27, 28], [28, 29], [29, 30],
        [31, 32], [32, 33], [33, 34], [34, 35],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
        [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66],
        [66, 67], [67, 60]
    ],
}


class Face(object):
    """
    The OpenPose face landmark detector model.

    Args:
        inference_size: set the size of the inference image size, suggested:
            368, 736, 1312, default 736
        gaussian_sigma: blur the heatmaps, default 2.5
        heatmap_peak_thresh: return landmark if over threshold, default 0.1

    """
    def __init__(self, face_model_path,
                 inference_size=None,
                 gaussian_sigma=None,
                 heatmap_peak_thresh=None):
        self.inference_size = inference_size or params["inference_img_size"]
        self.sigma = gaussian_sigma or params['gaussian_sigma']
        self.threshold = heatmap_peak_thresh or params["heatmap_peak_thresh"]
        self.model = FaceNet()
        self.model.load_state_dict(torch.load(face_model_path))
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
            # print('cuda')
        self.model.eval()

    def __call__(self, face_img):
        H, W, C = face_img.shape

        w_size = 384
        x_data = torch.from_numpy(util.smart_resize(face_img, (w_size, w_size))).permute([2, 0, 1]) / 256.0 - 0.5

        x_data = x_data.to(self.cn_device)

        with torch.no_grad():
            hs = self.model(x_data[None, ...])
            heatmaps = F.interpolate(
                hs[-1],
                (H, W),
                mode='bilinear', align_corners=True).cpu().numpy()[0]
        return heatmaps

    def compute_peaks_from_heatmaps(self, heatmaps):
        all_peaks = []
        for part in range(heatmaps.shape[0]):
            map_ori = heatmaps[part].copy()
            binary = np.ascontiguousarray(map_ori > 0.05, dtype=np.uint8)

            if np.sum(binary) == 0:
                continue

            positions = np.where(binary > 0.5)
            intensities = map_ori[positions]
            mi = np.argmax(intensities)
            y, x = positions[0][mi], positions[1][mi]
            all_peaks.append([x, y])

        return np.array(all_peaks)