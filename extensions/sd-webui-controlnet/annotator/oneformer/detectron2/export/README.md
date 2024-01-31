
This directory contains code to prepare a detectron2 model for deployment.
Currently it supports exporting a detectron2 model to TorchScript, ONNX, or (deprecated) Caffe2 format.

Please see [documentation](https://detectron2.readthedocs.io/tutorials/deployment.html) for its usage.


### Acknowledgements

Thanks to Mobile Vision team at Facebook for developing the Caffe2 conversion tools.

Thanks to Computing Platform Department - PAI team at Alibaba Group (@bddpqq, @chenbohua3) who
help export Detectron2 models to TorchScript.

Thanks to ONNX Converter team at Microsoft who help export Detectron2 models to ONNX.
