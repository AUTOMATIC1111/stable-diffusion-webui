import argparse

import onnx
from caffe2.python.onnx.backend import Caffe2Backend


parser = argparse.ArgumentParser(description="Convert ONNX to Caffe2")

parser.add_argument("model", help="The ONNX model")
parser.add_argument("--c2-prefix", required=True,
    help="The output file prefix for the caffe2 model init and predict file. ")


def main():
    args = parser.parse_args()
    onnx_model = onnx.load(args.model)
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    caffe2_init_str = caffe2_init.SerializeToString()
    with open(args.c2_prefix + '.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open(args.c2_prefix + '.predict.pb', "wb") as f:
        f.write(caffe2_predict_str)


if __name__ == "__main__":
    main()
