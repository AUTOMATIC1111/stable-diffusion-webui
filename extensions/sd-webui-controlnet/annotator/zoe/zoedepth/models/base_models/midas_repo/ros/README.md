# MiDaS for ROS1 by using LibTorch in C++

### Requirements

- Ubuntu 17.10 / 18.04 / 20.04, Debian Stretch
- ROS Melodic for Ubuntu (17.10 / 18.04) / Debian Stretch, ROS Noetic for Ubuntu 20.04
- C++11
- LibTorch >= 1.6

## Quick Start with a MiDaS Example

MiDaS is a neural network to compute depth from a single image.

* input from `image_topic`: `sensor_msgs/Image` - `RGB8` image with any shape
* output to `midas_topic`: `sensor_msgs/Image` - `TYPE_32FC1` inverse relative depth maps in range [0 - 255] with original size and channels=1

### Install Dependecies

* install ROS Melodic for Ubuntu 17.10 / 18.04:
```bash
wget https://raw.githubusercontent.com/isl-org/MiDaS/master/ros/additions/install_ros_melodic_ubuntu_17_18.sh
./install_ros_melodic_ubuntu_17_18.sh
```

or Noetic for Ubuntu 20.04: 

```bash
wget https://raw.githubusercontent.com/isl-org/MiDaS/master/ros/additions/install_ros_noetic_ubuntu_20.sh
./install_ros_noetic_ubuntu_20.sh
```


* install LibTorch 1.7 with CUDA 11.0:

On **Jetson (ARM)**:
```bash
wget https://nvidia.box.com/shared/static/wa34qwrwtk9njtyarwt5nvo6imenfy26.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy torch-1.7.0-cp36-cp36m-linux_aarch64.whl
```
Or compile LibTorch from source: https://github.com/pytorch/pytorch#from-source

On **Linux (x86_64)**:
```bash
cd ~/
wget https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcu110.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.7.0+cu110.zip
```

* create symlink for OpenCV:

```bash
sudo ln -s /usr/include/opencv4 /usr/include/opencv
```

* download and install MiDaS:

```bash
source ~/.bashrc
cd ~/
mkdir catkin_ws
cd catkin_ws
git clone https://github.com/isl-org/MiDaS
mkdir src
cp -r MiDaS/ros/* src

chmod +x src/additions/*.sh
chmod +x src/*.sh
chmod +x src/midas_cpp/scripts/*.py
cp src/additions/do_catkin_make.sh ./do_catkin_make.sh
./do_catkin_make.sh
./src/additions/downloads.sh
```

### Usage

* run only `midas` node: `~/catkin_ws/src/launch_midas_cpp.sh`

#### Test

* Test - capture video and show result in the window:
    * place any `test.mp4` video file to the directory `~/catkin_ws/src/`
    * run `midas` node: `~/catkin_ws/src/launch_midas_cpp.sh`
    * run test nodes in another terminal: `cd ~/catkin_ws/src && ./run_talker_listener_test.sh` and wait 30 seconds
    
    (to use Python 2, run command `sed -i 's/python3/python2/' ~/catkin_ws/src/midas_cpp/scripts/*.py` )

## Mobile version of MiDaS - Monocular Depth Estimation

### Accuracy

* MiDaS v2 small - ResNet50 default-decoder 384x384
* MiDaS v2.1 small - EfficientNet-Lite3 small-decoder 256x256

**Zero-shot error** (the lower - the better):

| Model |  DIW WHDR | Eth3d AbsRel | Sintel AbsRel | Kitti δ>1.25 | NyuDepthV2 δ>1.25 | TUM δ>1.25 |
|---|---|---|---|---|---|---|
| MiDaS v2 small 384x384 | **0.1248** | 0.1550 | **0.3300** | **21.81** | 15.73 | 17.00 |
| MiDaS v2.1 small 256x256 | 0.1344 | **0.1344** | 0.3370 | 29.27 | **13.43** | **14.53** |
| Relative improvement, % | -8 % | **+13 %** | -2 % | -34 % | **+15 %** | **+15 %** |

None of Train/Valid/Test subsets of datasets (DIW, Eth3d, Sintel, Kitti, NyuDepthV2, TUM) were not involved in Training or Fine Tuning.

### Inference speed (FPS) on nVidia GPU

Inference speed excluding pre and post processing, batch=1, **Frames Per Second** (the higher - the better):

| Model | Jetson Nano, FPS | RTX 2080Ti, FPS |
|---|---|---|
| MiDaS v2 small 384x384 | 1.6 | 117 |
| MiDaS v2.1 small 256x256 | 8.1 | 232 |
| SpeedUp, X times | **5x** | **2x** |

### Citation

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
