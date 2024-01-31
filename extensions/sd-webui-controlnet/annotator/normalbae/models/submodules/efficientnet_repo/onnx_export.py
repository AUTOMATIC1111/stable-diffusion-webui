""" ONNX export script

Export PyTorch models as ONNX graphs.

This export script originally started as an adaptation of code snippets found at
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

The default parameters work with PyTorch 1.6 and ONNX 1.7 and produce an optimal ONNX graph
for hosting in the ONNX runtime (see onnx_validate.py). To export an ONNX model compatible
with caffe2 (see caffe2_benchmark.py and caffe2_validate.py), the --keep-init and --aten-fallback
flags are currently required.

Older versions of PyTorch/ONNX (tested PyTorch 1.4, ONNX 1.5) do not need extra flags for
caffe2 compatibility, but they produce a model that isn't as fast running on ONNX runtime.

Most new release of PyTorch and ONNX cause some sort of breakage in the export / usage of ONNX models.
Please do your research and search ONNX and PyTorch issue tracker before asking me. Thanks.

Copyright 2020 Ross Wightman
"""
import argparse
import torch
import numpy as np

import onnx
import geffnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('output', metavar='ONNX_FILE',
                    help='output model filename')
parser.add_argument('--model', '-m', metavar='MODEL', default='mobilenetv3_large_100',
                    help='model architecture (default: mobilenetv3_large_100)')
parser.add_argument('--opset', type=int, default=10,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')


def main():
    args = parser.parse_args()

    args.pretrained = True
    if args.checkpoint:
        args.pretrained = False

    print("==> Creating PyTorch {} model".format(args.model))
    # NOTE exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers
    # for models using SAME padding
    model = geffnet.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        exportable=True)

    model.eval()

    example_input = torch.randn((args.batch_size, 3, args.img_size or 224, args.img_size or 224), requires_grad=True)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.
    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    model(example_input)

    print("==> Exporting model to ONNX format at '{}'".format(args.output))
    input_names = ["input0"]
    output_names = ["output0"]
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
    if args.dynamic_size:
        dynamic_axes['input0'][2] = 'height'
        dynamic_axes['input0'][3] = 'width'
    if args.aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    torch_out = torch.onnx._export(
        model, example_input, args.output, export_params=True, verbose=True, input_names=input_names,
        output_names=output_names, keep_initializers_as_inputs=args.keep_init, dynamic_axes=dynamic_axes,
        opset_version=args.opset, operator_export_type=export_type)

    print("==> Loading and checking exported model from '{}'".format(args.output))
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Passed")

    if args.keep_init and args.aten_fallback:
        import caffe2.python.onnx.backend as onnx_caffe2
        # Caffe2 loading only works properly in newer PyTorch/ONNX combos when
        # keep_initializers_as_inputs and aten_fallback are set to True.
        print("==> Loading model into Caffe2 backend and comparing forward pass.".format(args.output))
        caffe2_backend = onnx_caffe2.prepare(onnx_model)
        B = {onnx_model.graph.input[0].name: x.data.numpy()}
        c2_out = caffe2_backend.run(B)[0]
        np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)
        print("==> Passed")


if __name__ == '__main__':
    main()
