# (Generic) EfficientNets for PyTorch

A 'generic' implementation of EfficientNet, MixNet, MobileNetV3, etc. that covers most of the compute/parameter efficient architectures derived from the MobileNet V1/V2 block sequence, including those found via automated neural architecture search. 

All models are implemented by GenEfficientNet or MobileNetV3 classes, with string based architecture definitions to configure the block layouts (idea from [here](https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py))

## What's New

### Aug 19, 2020
* Add updated PyTorch trained EfficientNet-B3 weights trained by myself with `timm` (82.1 top-1)
* Add PyTorch trained EfficientNet-Lite0 contributed by [@hal-314](https://github.com/hal-314) (75.5 top-1)
* Update ONNX and Caffe2 export / utility scripts to work with latest PyTorch / ONNX
* ONNX runtime based validation script added
* activations (mostly) brought in sync with `timm` equivalents


### April 5, 2020
* Add some newly trained MobileNet-V2 models trained with latest h-params, rand augment. They compare quite favourably to EfficientNet-Lite
  * 3.5M param MobileNet-V2 100 @ 73%
  * 4.5M param MobileNet-V2 110d @ 75%
  * 6.1M param MobileNet-V2 140 @ 76.5%
  * 5.8M param MobileNet-V2 120d @ 77.3%
  
### March 23, 2020
 * Add EfficientNet-Lite models w/ weights ported from [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
 * Add PyTorch trained MobileNet-V3 Large weights with 75.77% top-1
 * IMPORTANT CHANGE (if training from scratch) - weight init changed to better match Tensorflow impl, set `fix_group_fanout=False` in `initialize_weight_goog` for old behavior

### Feb 12, 2020
 * Add EfficientNet-L2 and B0-B7 NoisyStudent weights ported from [Tensorflow TPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
 * Port new EfficientNet-B8 (RandAugment) weights from TF TPU, these are different than the B8 AdvProp, different input normalization.
 * Add RandAugment PyTorch trained EfficientNet-ES (EdgeTPU-Small) weights with 78.1 top-1. Trained by [Andrew Lavin](https://github.com/andravin)

### Jan 22, 2020
 * Update weights for EfficientNet B0, B2, B3 and MixNet-XL with latest RandAugment trained weights. Trained with (https://github.com/rwightman/pytorch-image-models)
 * Fix torchscript compatibility for PyTorch 1.4, add torchscript support for MixedConv2d using ModuleDict
 * Test models, torchscript, onnx export with PyTorch 1.4 -- no issues

### Nov 22, 2019
 * New top-1 high! Ported official TF EfficientNet AdvProp (https://arxiv.org/abs/1911.09665) weights and B8 model spec. Created a new set of `ap` models since they use a different
 preprocessing (Inception mean/std) from the original EfficientNet base/AA/RA weights.
 
### Nov 15, 2019
 * Ported official TF MobileNet-V3 float32 large/small/minimalistic weights
 * Modifications to MobileNet-V3 model and components to support some additional config needed for differences between TF MobileNet-V3 and mine

### Oct 30, 2019
 * Many of the models will now work with torch.jit.script, MixNet being the biggest exception
 * Improved interface for enabling torchscript or ONNX export compatible modes (via config)
 * Add JIT optimized mem-efficient Swish/Mish autograd.fn in addition to memory-efficient autgrad.fn
 * Activation factory to select best version of activation by name or override one globally
 * Add pretrained checkpoint load helper that handles input conv and classifier changes
 
### Oct 27, 2019
 * Add CondConv EfficientNet variants ported from https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
 * Add RandAug weights for TF EfficientNet B5 and B7 from https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
 * Bring over MixNet-XL model and depth scaling algo from my pytorch-image-models code base
 * Switch activations and global pooling to modules
 * Add memory-efficient Swish/Mish impl
 * Add as_sequential() method to all models and allow as an argument in entrypoint fns
 * Move MobileNetV3 into own file since it has a different head
 * Remove ChamNet, MobileNet V2/V1 since they will likely never be used here

## Models

Implemented models include:
  * EfficientNet NoisyStudent (B0-B7, L2) (https://arxiv.org/abs/1911.04252)
  * EfficientNet AdvProp (B0-B8) (https://arxiv.org/abs/1911.09665)
  * EfficientNet (B0-B8) (https://arxiv.org/abs/1905.11946)
  * EfficientNet-EdgeTPU (S, M, L) (https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
  * EfficientNet-CondConv (https://arxiv.org/abs/1904.04971)
  * EfficientNet-Lite (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
  * MixNet (https://arxiv.org/abs/1907.09595)
  * MNASNet B1, A1 (Squeeze-Excite), and Small (https://arxiv.org/abs/1807.11626)
  * MobileNet-V3 (https://arxiv.org/abs/1905.02244)
  * FBNet-C (https://arxiv.org/abs/1812.03443)
  * Single-Path NAS (https://arxiv.org/abs/1904.02877)
    
I originally implemented and trained some these models with code [here](https://github.com/rwightman/pytorch-image-models), this repository contains just the GenEfficientNet models, validation, and associated ONNX/Caffe2 export code. 

## Pretrained

I've managed to train several of the models to accuracies close to or above the originating papers and official impl. My training code is here: https://github.com/rwightman/pytorch-image-models


|Model | Prec@1 (Err) | Prec@5 (Err) | Param#(M) | MAdds(M) | Image Scaling | Resolution | Crop |
|---|---|---|---|---|---|---|---|
| efficientnet_b3 | 82.240 (17.760) | 96.116 (3.884) | 12.23 | TBD | bicubic | 320 | 1.0 |
| efficientnet_b3 | 82.076 (17.924) | 96.020 (3.980) | 12.23 | TBD | bicubic | 300 | 0.904 |
| mixnet_xl | 81.074 (18.926) | 95.282 (4.718) | 11.90 | TBD | bicubic | 256 | 1.0 |
| efficientnet_b2 | 80.612 (19.388) | 95.318 (4.682) | 9.1 | TBD | bicubic | 288 | 1.0 |
| mixnet_xl | 80.476 (19.524) | 94.936 (5.064) | 11.90 | TBD | bicubic | 224 | 0.875 |
| efficientnet_b2 | 80.288 (19.712) | 95.166 (4.834) | 9.1 | 1003 | bicubic | 260 | 0.890 |
| mixnet_l | 78.976 (21.024 | 94.184 (5.816) | 7.33 | TBD | bicubic | 224 | 0.875 |
| efficientnet_b1 | 78.692 (21.308) | 94.086 (5.914) | 7.8 | 694 | bicubic | 240 | 0.882 |
| efficientnet_es | 78.066 (21.934) | 93.926 (6.074) | 5.44 | TBD | bicubic | 224 | 0.875 |
| efficientnet_b0 | 77.698 (22.302) | 93.532 (6.468) | 5.3 | 390 | bicubic | 224 | 0.875 |
| mobilenetv2_120d | 77.294 (22.706 | 93.502 (6.498) | 5.8 | TBD | bicubic | 224 | 0.875 |
| mixnet_m | 77.256 (22.744) | 93.418 (6.582) | 5.01 | 353 | bicubic | 224 | 0.875 |
| mobilenetv2_140 | 76.524 (23.476) | 92.990 (7.010) | 6.1 | TBD | bicubic | 224 | 0.875 |
| mixnet_s | 75.988 (24.012) | 92.794 (7.206) | 4.13 | TBD | bicubic | 224 | 0.875 |
| mobilenetv3_large_100 | 75.766 (24.234) | 92.542 (7.458) | 5.5 | TBD | bicubic | 224 | 0.875 |
| mobilenetv3_rw | 75.634 (24.366) | 92.708 (7.292) | 5.5 | 219 | bicubic | 224 | 0.875 |
| efficientnet_lite0 | 75.472 (24.528) | 92.520 (7.480) | 4.65 | TBD | bicubic | 224 | 0.875 |
| mnasnet_a1 | 75.448 (24.552) | 92.604 (7.396) | 3.9 | 312 | bicubic | 224 | 0.875 |
| fbnetc_100 | 75.124 (24.876) | 92.386 (7.614) | 5.6 | 385 | bilinear | 224 | 0.875 |
| mobilenetv2_110d | 75.052 (24.948) | 92.180 (7.820) | 4.5 | TBD | bicubic | 224 | 0.875 |
| mnasnet_b1 | 74.658 (25.342) | 92.114 (7.886) | 4.4 | 315 | bicubic | 224 | 0.875 |
| spnasnet_100 | 74.084 (25.916)  | 91.818 (8.182) | 4.4 | TBD | bilinear | 224 | 0.875 |
| mobilenetv2_100 | 72.978 (27.022) | 91.016 (8.984) | 3.5 | TBD | bicubic | 224 | 0.875 |


More pretrained models to come...


## Ported Weights

The weights ported from Tensorflow checkpoints for the EfficientNet models do pretty much match accuracy in Tensorflow once a SAME convolution padding equivalent is added, and the same crop factors, image scaling, etc (see table) are used via cmd line args.

**IMPORTANT:** 
* Tensorflow ported weights for EfficientNet AdvProp (AP), EfficientNet EdgeTPU, EfficientNet-CondConv, EfficientNet-Lite, and MobileNet-V3 models use Inception style (0.5, 0.5, 0.5) for mean and std.
* Enabling the Tensorflow preprocessing pipeline with `--tf-preprocessing` at validation time will improve scores by 0.1-0.5%, very close to original TF impl.

To run validation for tf_efficientnet_b5:
`python validate.py /path/to/imagenet/validation/ --model tf_efficientnet_b5 -b 64 --img-size 456 --crop-pct 0.934 --interpolation bicubic`

To run validation w/ TF preprocessing for tf_efficientnet_b5:
`python validate.py /path/to/imagenet/validation/ --model tf_efficientnet_b5 -b 64 --img-size 456 --tf-preprocessing`

To run validation for a model with Inception preprocessing, ie EfficientNet-B8 AdvProp:
`python validate.py /path/to/imagenet/validation/ --model tf_efficientnet_b8_ap -b 48 --num-gpu 2 --img-size 672 --crop-pct 0.954 --mean 0.5 --std 0.5`

|Model | Prec@1 (Err) | Prec@5 (Err) | Param # | Image Scaling  | Image Size | Crop | 
|---|---|---|---|---|---|---|
| tf_efficientnet_l2_ns *tfp | 88.352 (11.648) | 98.652 (1.348) | 480 | bicubic | 800 | N/A |
| tf_efficientnet_l2_ns      | TBD | TBD | 480 | bicubic | 800 | 0.961 |
| tf_efficientnet_l2_ns_475      | 88.234 (11.766) | 98.546 (1.454) | 480 | bicubic | 475 | 0.936 |
| tf_efficientnet_l2_ns_475 *tfp | 88.172 (11.828) | 98.566 (1.434) | 480 | bicubic | 475 | N/A |
| tf_efficientnet_b7_ns *tfp | 86.844 (13.156) | 98.084 (1.916) | 66.35 | bicubic | 600 | N/A |
| tf_efficientnet_b7_ns      | 86.840 (13.160) | 98.094 (1.906) | 66.35 | bicubic | 600 | N/A |
| tf_efficientnet_b6_ns      | 86.452 (13.548) | 97.882 (2.118) | 43.04 | bicubic | 528 | N/A |
| tf_efficientnet_b6_ns *tfp | 86.444 (13.556) | 97.880 (2.120) | 43.04 | bicubic | 528 | N/A |
| tf_efficientnet_b5_ns *tfp | 86.064 (13.936) | 97.746 (2.254) | 30.39 | bicubic | 456 | N/A |
| tf_efficientnet_b5_ns      | 86.088 (13.912) | 97.752 (2.248) | 30.39 | bicubic | 456 | N/A |
| tf_efficientnet_b8_ap *tfp | 85.436 (14.564) | 97.272 (2.728) | 87.4 | bicubic | 672 | N/A |
| tf_efficientnet_b8 *tfp    | 85.384 (14.616) | 97.394 (2.606) | 87.4 | bicubic | 672 | N/A |
| tf_efficientnet_b8         | 85.370 (14.630) | 97.390 (2.610) | 87.4 | bicubic | 672 | 0.954 |
| tf_efficientnet_b8_ap      | 85.368 (14.632) | 97.294 (2.706) | 87.4 | bicubic | 672 | 0.954 |
| tf_efficientnet_b4_ns *tfp | 85.298 (14.702) | 97.504 (2.496) | 19.34 | bicubic | 380 | N/A |
| tf_efficientnet_b4_ns      | 85.162 (14.838) | 97.470 (2.530) | 19.34 | bicubic | 380 | 0.922 |
| tf_efficientnet_b7_ap *tfp | 85.154 (14.846) | 97.244 (2.756) | 66.35 | bicubic | 600 | N/A |
| tf_efficientnet_b7_ap      | 85.118 (14.882) | 97.252 (2.748) | 66.35 | bicubic | 600 | 0.949 |
| tf_efficientnet_b7 *tfp | 84.940 (15.060) | 97.214 (2.786) | 66.35 | bicubic | 600 | N/A |
| tf_efficientnet_b7      | 84.932 (15.068) | 97.208 (2.792) | 66.35 | bicubic | 600 | 0.949 |
| tf_efficientnet_b6_ap      | 84.786 (15.214) | 97.138 (2.862) | 43.04 | bicubic | 528 | 0.942 |
| tf_efficientnet_b6_ap *tfp | 84.760 (15.240) | 97.124 (2.876) | 43.04 | bicubic | 528 | N/A |
| tf_efficientnet_b5_ap *tfp | 84.276 (15.724) | 96.932 (3.068) | 30.39 | bicubic | 456 | N/A |
| tf_efficientnet_b5_ap      | 84.254 (15.746) | 96.976 (3.024) | 30.39 | bicubic | 456 | 0.934 |
| tf_efficientnet_b6 *tfp  | 84.140 (15.860) | 96.852 (3.148) | 43.04 | bicubic | 528 | N/A |
| tf_efficientnet_b6       | 84.110 (15.890) | 96.886 (3.114) | 43.04 | bicubic | 528 | 0.942 |
| tf_efficientnet_b3_ns *tfp | 84.054 (15.946) | 96.918 (3.082) | 12.23 | bicubic | 300 | N/A |
| tf_efficientnet_b3_ns      | 84.048 (15.952) | 96.910 (3.090) | 12.23 | bicubic | 300 | .904 |
| tf_efficientnet_b5 *tfp  | 83.822 (16.178) | 96.756 (3.244) | 30.39 | bicubic | 456 | N/A |
| tf_efficientnet_b5       | 83.812 (16.188) | 96.748 (3.252) | 30.39 | bicubic | 456 | 0.934 |
| tf_efficientnet_b4_ap *tfp | 83.278 (16.722) | 96.376 (3.624) | 19.34 | bicubic | 380 | N/A |
| tf_efficientnet_b4_ap      | 83.248 (16.752) | 96.388 (3.612) | 19.34 | bicubic | 380 | 0.922 |
| tf_efficientnet_b4       | 83.022 (16.978) | 96.300 (3.700) | 19.34 | bicubic | 380 | 0.922 |
| tf_efficientnet_b4 *tfp  | 82.948 (17.052) | 96.308 (3.692) | 19.34 | bicubic | 380 | N/A |
| tf_efficientnet_b2_ns *tfp | 82.436 (17.564) | 96.268 (3.732) | 9.11 | bicubic | 260 | N/A |
| tf_efficientnet_b2_ns      | 82.380 (17.620) | 96.248 (3.752) | 9.11 | bicubic | 260 | 0.89 |
| tf_efficientnet_b3_ap *tfp | 81.882 (18.118) | 95.662 (4.338) | 12.23 | bicubic | 300 | N/A |
| tf_efficientnet_b3_ap      | 81.828 (18.172) | 95.624 (4.376) | 12.23 | bicubic | 300 | 0.904 |
| tf_efficientnet_b3       | 81.636 (18.364) | 95.718 (4.282) | 12.23 | bicubic | 300 | 0.904 |
| tf_efficientnet_b3 *tfp  | 81.576 (18.424) | 95.662 (4.338) | 12.23 | bicubic | 300 | N/A |
| tf_efficientnet_lite4      | 81.528 (18.472) | 95.668 (4.332) | 13.00  | bilinear | 380 | 0.92 |
| tf_efficientnet_b1_ns *tfp | 81.514 (18.486) | 95.776 (4.224) | 7.79 | bicubic | 240 | N/A |
| tf_efficientnet_lite4 *tfp | 81.502 (18.498) | 95.676 (4.324) | 13.00  | bilinear | 380 | N/A |
| tf_efficientnet_b1_ns      | 81.388 (18.612) | 95.738 (4.262) | 7.79 | bicubic | 240 | 0.88 |
| tf_efficientnet_el       | 80.534 (19.466) | 95.190 (4.810) | 10.59 | bicubic | 300 | 0.904 |
| tf_efficientnet_el *tfp  | 80.476 (19.524) | 95.200 (4.800) | 10.59 | bicubic | 300 | N/A |
| tf_efficientnet_b2_ap *tfp | 80.420 (19.580) | 95.040 (4.960) | 9.11 | bicubic | 260 | N/A |
| tf_efficientnet_b2_ap    | 80.306 (19.694) | 95.028 (4.972) | 9.11 | bicubic | 260 | 0.890 |
| tf_efficientnet_b2 *tfp  | 80.188 (19.812) | 94.974 (5.026) | 9.11 | bicubic | 260 | N/A |
| tf_efficientnet_b2       | 80.086 (19.914) | 94.908 (5.092) | 9.11 | bicubic | 260 | 0.890 |
| tf_efficientnet_lite3       | 79.812 (20.188) | 94.914 (5.086) | 8.20  | bilinear | 300 | 0.904 |
| tf_efficientnet_lite3 *tfp  | 79.734 (20.266) | 94.838 (5.162) | 8.20  | bilinear | 300 | N/A |
| tf_efficientnet_b1_ap *tfp | 79.532 (20.468) | 94.378 (5.622) | 7.79 | bicubic | 240 | N/A |
| tf_efficientnet_cc_b1_8e *tfp | 79.464 (20.536)| 94.492 (5.508) | 39.7 | bicubic | 240 | 0.88 |
| tf_efficientnet_cc_b1_8e | 79.298 (20.702) | 94.364 (5.636) | 39.7 | bicubic | 240 | 0.88 |
| tf_efficientnet_b1_ap    | 79.278 (20.722) | 94.308 (5.692) | 7.79 | bicubic | 240 | 0.88 |
| tf_efficientnet_b1 *tfp  | 79.172 (20.828) | 94.450 (5.550) | 7.79 | bicubic | 240 | N/A |
| tf_efficientnet_em *tfp  | 78.958 (21.042) | 94.458 (5.542) | 6.90 | bicubic | 240 | N/A |
| tf_efficientnet_b0_ns *tfp | 78.806 (21.194) | 94.496 (5.504) | 5.29 | bicubic | 224 | N/A |
| tf_mixnet_l *tfp         | 78.846 (21.154) | 94.212 (5.788) | 7.33 | bilinear | 224 | N/A |
| tf_efficientnet_b1       | 78.826 (21.174) | 94.198 (5.802) | 7.79 | bicubic | 240 | 0.88 |
| tf_mixnet_l              | 78.770 (21.230) | 94.004 (5.996) | 7.33 | bicubic | 224 | 0.875 |
| tf_efficientnet_em       | 78.742 (21.258) | 94.332 (5.668) | 6.90 | bicubic | 240 | 0.875 |
| tf_efficientnet_b0_ns    | 78.658 (21.342) | 94.376 (5.624) | 5.29 | bicubic | 224 | 0.875 |
| tf_efficientnet_cc_b0_8e *tfp | 78.314 (21.686) | 93.790 (6.210) | 24.0 | bicubic | 224 | 0.875 |
| tf_efficientnet_cc_b0_8e | 77.908 (22.092) | 93.656 (6.344) | 24.0 | bicubic | 224 | 0.875 |
| tf_efficientnet_cc_b0_4e *tfp | 77.746 (22.254) | 93.552 (6.448) | 13.3 | bicubic | 224 | 0.875 |
| tf_efficientnet_cc_b0_4e | 77.304 (22.696) | 93.332 (6.668) | 13.3 | bicubic | 224 | 0.875 |
| tf_efficientnet_es *tfp  | 77.616 (22.384) | 93.750 (6.250) | 5.44 | bicubic | 224 | N/A |
| tf_efficientnet_lite2 *tfp  | 77.544 (22.456) | 93.800 (6.200) | 6.09  | bilinear | 260 | N/A |
| tf_efficientnet_lite2       | 77.460 (22.540) | 93.746 (6.254) | 6.09  | bicubic | 260 | 0.89 |
| tf_efficientnet_b0_ap *tfp | 77.514 (22.486) | 93.576 (6.424) | 5.29  | bicubic | 224 | N/A |
| tf_efficientnet_es       | 77.264 (22.736) | 93.600 (6.400) | 5.44 | bicubic | 224 | N/A |
| tf_efficientnet_b0 *tfp  | 77.258 (22.742) | 93.478 (6.522) | 5.29  | bicubic | 224 | N/A |
| tf_efficientnet_b0_ap    | 77.084 (22.916) | 93.254 (6.746) | 5.29  | bicubic | 224 | 0.875 |
| tf_mixnet_m *tfp         | 77.072 (22.928) | 93.368 (6.632) | 5.01 | bilinear | 224 | N/A |
| tf_mixnet_m              | 76.950 (23.050) | 93.156 (6.844) | 5.01 | bicubic | 224 | 0.875 |
| tf_efficientnet_b0       | 76.848 (23.152) | 93.228 (6.772) | 5.29  | bicubic | 224 | 0.875 |
| tf_efficientnet_lite1 *tfp  | 76.764 (23.236) | 93.326 (6.674) | 5.42  | bilinear | 240 | N/A |
| tf_efficientnet_lite1       | 76.638 (23.362) | 93.232 (6.768) | 5.42  | bicubic | 240 | 0.882 |
| tf_mixnet_s *tfp         | 75.800 (24.200) | 92.788 (7.212) | 4.13 | bilinear | 224 | N/A |
| tf_mobilenetv3_large_100 *tfp | 75.768 (24.232) | 92.710 (7.290) | 5.48 | bilinear | 224 | N/A |
| tf_mixnet_s              | 75.648 (24.352) | 92.636 (7.364) | 4.13 | bicubic | 224 | 0.875 |
| tf_mobilenetv3_large_100 | 75.516 (24.484) | 92.600 (7.400) | 5.48 | bilinear | 224 | 0.875 |
| tf_efficientnet_lite0 *tfp  | 75.074 (24.926) | 92.314 (7.686) | 4.65  | bilinear | 224 | N/A |
| tf_efficientnet_lite0       | 74.842 (25.158) | 92.170 (7.830) | 4.65  | bicubic | 224 | 0.875 |
| tf_mobilenetv3_large_075 *tfp | 73.730 (26.270) | 91.616 (8.384) | 3.99 | bilinear | 224 |N/A |
| tf_mobilenetv3_large_075 | 73.442 (26.558) | 91.352 (8.648) | 3.99 | bilinear | 224 | 0.875 |
| tf_mobilenetv3_large_minimal_100 *tfp | 72.678 (27.322) | 90.860 (9.140) | 3.92 | bilinear | 224 | N/A |
| tf_mobilenetv3_large_minimal_100 | 72.244 (27.756) | 90.636 (9.364) | 3.92 | bilinear | 224 | 0.875 |
| tf_mobilenetv3_small_100 *tfp | 67.918 (32.082) | 87.958 (12.042 | 2.54 | bilinear | 224 | N/A |
| tf_mobilenetv3_small_100 | 67.918 (32.082) | 87.662 (12.338) | 2.54 | bilinear | 224 | 0.875 |
| tf_mobilenetv3_small_075 *tfp | 66.142 (33.858) | 86.498 (13.502) | 2.04 | bilinear | 224 | N/A |
| tf_mobilenetv3_small_075 | 65.718 (34.282) | 86.136 (13.864) | 2.04 | bilinear | 224 | 0.875 |
| tf_mobilenetv3_small_minimal_100 *tfp | 63.378 (36.622) | 84.802 (15.198) | 2.04 | bilinear | 224 | N/A |
| tf_mobilenetv3_small_minimal_100 | 62.898 (37.102) | 84.230 (15.770) | 2.04 | bilinear | 224 | 0.875 |


*tfp models validated with `tf-preprocessing` pipeline

Google tf and tflite weights ported from official Tensorflow repositories
* https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
* https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
* https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet

## Usage

### Environment

All development and testing has been done in Conda Python 3 environments on Linux x86-64 systems, specifically Python 3.6.x, 3.7.x, 3.8.x.

Users have reported that a Python 3 Anaconda install in Windows works. I have not verified this myself.

PyTorch versions 1.4, 1.5, 1.6 have been tested with this code.

I've tried to keep the dependencies minimal, the setup is as per the PyTorch default install instructions for Conda:
```
conda create -n torch-env
conda activate torch-env
conda install -c pytorch pytorch torchvision cudatoolkit=10.2
```

### PyTorch Hub

Models can be accessed via the PyTorch Hub API

```
>>> torch.hub.list('rwightman/gen-efficientnet-pytorch')
['efficientnet_b0', ...]
>>> model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
>>> model.eval()
>>> output = model(torch.randn(1,3,224,224))
```

### Pip
This package can be installed via pip.

Install (after conda env/install):
```
pip install geffnet
```

Eval use:
```
>>> import geffnet
>>> m = geffnet.create_model('mobilenetv3_large_100', pretrained=True)
>>> m.eval()
```

Train use:
```
>>> import geffnet
>>> # models can also be created by using the entrypoint directly
>>> m = geffnet.efficientnet_b2(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)
>>> m.train()
```

Create in a nn.Sequential container, for fast.ai, etc:
```
>>> import geffnet
>>> m = geffnet.mixnet_l(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2, as_sequential=True)
```

### Exporting

Scripts are included to
* export models to ONNX (`onnx_export.py`)
* optimized ONNX graph (`onnx_optimize.py` or `onnx_validate.py` w/ `--onnx-output-opt` arg)
* validate with ONNX runtime  (`onnx_validate.py`)
* convert ONNX model to Caffe2 (`onnx_to_caffe.py`)
* validate in Caffe2 (`caffe2_validate.py`)
* benchmark in Caffe2 w/ FLOPs, parameters output (`caffe2_benchmark.py`)

As an example, to export the MobileNet-V3 pretrained model and then run an Imagenet validation:
```
python onnx_export.py --model mobilenetv3_large_100 ./mobilenetv3_100.onnx
python onnx_validate.py /imagenet/validation/ --onnx-input ./mobilenetv3_100.onnx 
```

These scripts were tested to be working as of PyTorch 1.6 and ONNX 1.7 w/ ONNX runtime 1.4. Caffe2 compatible
export now requires additional args mentioned in the export script (not needed in earlier versions).

#### Export Notes
1. The TF ported weights with the 'SAME' conv padding activated cannot be exported to ONNX unless `_EXPORTABLE` flag in `config.py` is set to True. Use `config.set_exportable(True)` as in the `onnx_export.py` script.
2. TF ported models with 'SAME' padding will have the padding fixed at export time to the resolution used for export. Even though dynamic padding is supported in opset >= 11, I can't get it working.
3. ONNX optimize facility doesn't work reliably in PyTorch 1.6 / ONNX 1.7. Fortunately, the onnxruntime based inference is working very well now and includes on the fly optimization.
3. ONNX / Caffe2 export/import frequently breaks with different PyTorch and ONNX version releases. Please check their respective issue trackers before filing issues here.


