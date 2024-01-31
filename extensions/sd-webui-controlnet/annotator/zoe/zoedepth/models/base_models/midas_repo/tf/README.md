## Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

### TensorFlow inference using `.pb` and `.onnx` models

1. [Run inference on TensorFlow-model by using TensorFlow](#run-inference-on-tensorflow-model-by-using-tensorFlow)

2. [Run inference on ONNX-model by using TensorFlow](#run-inference-on-onnx-model-by-using-tensorflow)

3. [Make ONNX model from downloaded Pytorch model file](#make-onnx-model-from-downloaded-pytorch-model-file)


### Run inference on TensorFlow-model by using TensorFlow

1) Download the model weights [model-f6b98070.pb](https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.pb) 
and [model-small.pb](https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.pb) and place the
file in the `/tf/` folder.

2) Set up dependencies: 

```shell
# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install TensorFlow
pip install -I grpcio tensorflow==2.3.0 tensorflow-addons==0.11.2 numpy==1.18.0
```

#### Usage

1) Place one or more input images in the folder `tf/input`.

2) Run the model:

    ```shell
    python tf/run_pb.py
    ```

    Or run the small model:

    ```shell
    python tf/run_pb.py --model_weights model-small.pb --model_type small
    ```

3) The resulting inverse depth maps are written to the `tf/output` folder.


### Run inference on ONNX-model by using ONNX-Runtime

1) Download the model weights [model-f6b98070.onnx](https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx) 
and [model-small.onnx](https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx) and place the
file in the `/tf/` folder.

2) Set up dependencies: 

```shell
# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install ONNX
pip install onnx==1.7.0

# install ONNX Runtime
pip install onnxruntime==1.5.2
```

#### Usage

1) Place one or more input images in the folder `tf/input`.

2) Run the model:

    ```shell
    python tf/run_onnx.py
    ```

    Or run the small model:

    ```shell
    python tf/run_onnx.py --model_weights model-small.onnx --model_type small
    ```

3) The resulting inverse depth maps are written to the `tf/output` folder.



### Make ONNX model from downloaded Pytorch model file

1) Download the model weights [model-f6b98070.pt](https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.pt) and place the
file in the root folder.

2) Set up dependencies: 

```shell
# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install PyTorch TorchVision
pip install -I torch==1.7.0 torchvision==0.8.0

# install TensorFlow
pip install -I grpcio tensorflow==2.3.0 tensorflow-addons==0.11.2 numpy==1.18.0

# install ONNX
pip install onnx==1.7.0

# install ONNX-TensorFlow
git clone https://github.com/onnx/onnx-tensorflow.git
cd onnx-tensorflow 
git checkout 095b51b88e35c4001d70f15f80f31014b592b81e 
pip install -e .
```

#### Usage

1) Run the converter:

    ```shell
    python tf/make_onnx_model.py
    ```

2) The resulting `model-f6b98070.onnx` file is written to the `/tf/` folder.


### Requirements

   The code was tested with Python 3.6.9, PyTorch 1.5.1, TensorFlow 2.2.0, TensorFlow-addons 0.8.3, ONNX 1.7.0, ONNX-TensorFlow (GitHub-master-17.07.2020) and OpenCV 4.3.0.
 
### Citation

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2019,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

### License 

MIT License 

   
