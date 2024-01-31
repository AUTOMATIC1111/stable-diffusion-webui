from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from contextlib import suppress

import geffnet
from data import Dataset, create_loader, resolve_data_config
from utils import accuracy, AverageMeter

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='spnasnet1_00',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--crop-pct', type=float, default=None, metavar='PCT',
                    help='Override default crop pct of 0.875')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--tf-preprocessing', dest='tf_preprocessing', action='store_true',
                    help='use tensorflow mnasnet preporcessing')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true',
                    help='')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use native Torch AMP mixed precision.')


def main():
    args = parser.parse_args()

    if not args.checkpoint and not args.pretrained:
        args.pretrained = True

    amp_autocast = suppress  # do nothing
    if args.amp:
        if not has_native_amp:
            print("Native Torch AMP is not available (requires torch >= 1.6), using FP32.")
        else:
            amp_autocast = torch.cuda.amp.autocast

    # create model
    model = geffnet.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        scriptable=args.torchscript)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    print('Model %s created, param count: %d' %
          (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(model, args)

    criterion = nn.CrossEntropyLoss()

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()

    loader = create_loader(
        Dataset(args.data, load_bytes=args.tf_preprocessing),
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=not args.no_cuda,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        tensorflow_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            if not args.no_cuda:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s) \t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    rate_avg=input.size(0) / batch_time.avg,
                    loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} ({top1a:.3f}) Prec@5 {top5.avg:.3f} ({top5a:.3f})'.format(
        top1=top1, top1a=100-top1.avg, top5=top5, top5a=100.-top5.avg))


if __name__ == '__main__':
    main()
