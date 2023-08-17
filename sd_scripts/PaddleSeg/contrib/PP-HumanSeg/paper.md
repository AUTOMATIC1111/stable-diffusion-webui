# Semantic Connectivity-aware Segmentation With a Large-Scale Teleconferencing Video Dataset
Official resource for the paper *PP-HumanSeg: Connectivity-Aware Portrait Segmentation With a Large-Scale Teleconferencing Video Dataset*. [[Paper](https://arxiv.org/abs/2112.07146) | [Poster](https://paddleseg.bj.bcebos.com/dygraph/humanseg/paper/12-HAD-poster.pdf) | [YouTube](https://www.youtube.com/watch?v=FlK8R5cdD7E)]

## Semantic Connectivity-aware Learning
SCL (Semantic Connectivity-aware Learning) framework, which introduces a SC Loss (Semantic Connectivity-aware Loss) to improve the quality of segmentation results from the perspective of connectivity. SCL can improve the integrity of segmentation objects and increase segmentation accuracy. Support multi-class segmentation. [[Source code](../../paddleseg/models/losses/semantic_connectivity_loss.py)]

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148921096-29a4f90f-2113-4f97-87b5-19364e83b454.png" width="40%" height="40%">
</p>

### Connected Components Calculation and Matching
<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148931627-bfaeeecb-c260-4d00-9393-a7e52a56ce18.png" width="40%" height="40%">
</p>
(a) It indicates prediction and ground truth, i.e. P and G. (b) Connected components are generated through the CCL algorithm, respectively. (c) Connected components are matched using the IoU value.

### Segmentation Results

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148931612-bfc5a7f2-f6b7-4666-b2dd-86926ea7bfd7.png" width="60%" height="60%">
</p>


### Perfermance on Cityscapes
The experimental results on our Teleconferencing Video Dataset are shown in paper, and the experimental results on Cityscapes are as follows:

| Model | Backbone | Learning Strategy | GPUs * Batch Size(Per Card)| Training Iters | mIoU (%) | Config |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|OCRNet|HRNet-W48|-|2*2|40000|76.23| [config](../../configs/ocrnet/ocrnet_hrnetw48_cityscapes_1024x512_40k.yml) |
|OCRNet|HRNet-W48|SCL|2*2|40000|78.29(**+2.06**)|[config](../../configs/ocrnet/ocrnet_hrnetw48_cityscapes_1024x512_40k_SCL.yml) |
|FCN|HRNet-W18|-|2*4|80000|77.81|[config](../../configs/fcn/fcn_hrnetw18_cityscapes_1024x512_80k_bs4.yml)|
|FCN|HRNet-W18|SCL|2*4|80000|78.68(**+0.87**)|[config](../../configs/fcn/fcn_hrnetw18_cityscapes_1024x512_80k_bs4_SCL.yml)|
|Fast SCNN|-|-|2*4|40000|56.41|[config](../../configs/fastscnn/fastscnn_cityscapes_1024x1024_40k.yml)|
|Fast SCNN|-|SCL|2*4|40000|57.37(**+0.96**)|[config](../../configs/fastscnn/fastscnn_cityscapes_1024x1024_40k_SCL.yml)|



## PP-HumanSeg14K: A Large-Scale Teleconferencing Video Dataset
A large-scale video portrait dataset that contains 291 videos from 23 conference scenes with 14K frames. This dataset contains various teleconferencing scenes, various actions of the participants, interference of passers-by and illumination change. The data can be obtained by sending an email to paddleseg@baidu.com via an **official email** (not use qq, gmail, etc.) including your institution/company information and the purpose on the dataset.

<p align="center">
<img src="https://user-images.githubusercontent.com/30695251/148931684-cc10c994-3bd4-4d0c-9bcc-283f9bbc6ac9.png" width="80%" height="80%">
</p>

## Citation
If our project is useful in your research, please citing:

```latex
@InProceedings{Chu_2022_WACV,
    author    = {Chu, Lutao and Liu, Yi and Wu, Zewu and Tang, Shiyu and Chen, Guowei and Hao, Yuying and Peng, Juncai and Yu, Zhiliang and Chen, Zeyu and Lai, Baohua and Xiong, Haoyi},
    title     = {PP-HumanSeg: Connectivity-Aware Portrait Segmentation With a Large-Scale Teleconferencing Video Dataset},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {202-209}
}
```
