import os
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from loguru import logger
import glob
import tqdm

import sys
sys.path.append("handlers/rmf")

from .dataloader_collect import RescaleT
from .dataloader_collect import ToTensor
from .dataloader_collect import SalObjDataset

from .model import myNet1
from myconfig import myParser
from PIL import Image

from modules import shared, scripts

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output_cv2(image_name, pred, d_dir, origin_shape):
    predict = pred.cpu().numpy().transpose((1, 2, 0))
    im = predict
    import numpy as np
    imo = cv2.resize(im, (int(origin_shape[1]), int(origin_shape[0])))[
        :, :, np.newaxis]  # mask
    import numpy as np
    from PIL import Image
    mask = imo
    img_array = np.array(Image.open(image_name))
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = np.concatenate((mask * img + 1 - mask, mask *
                         255), axis=2).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img

def rmf_seg(path, net=None):

    logger.info("rmf seg begining.....")
    args = myParser()
    pretrained_path = os.path.join(scripts.basedir(), "models" + os.path.sep + "RMFormer" +
                              os.path.sep + "pretrain"+ os.path.sep + "swin_base_patch4_window12_384_22k.pth")
    model_dir=os.path.join(scripts.basedir(), "models" + os.path.sep + "RMFormer" +
                              os.path.sep + "Atemp"+ os.path.sep + "model_DH_final.pth")
    args.pretrained_path=pretrained_path
    net = myNet1(args)
    net.load_state_dict(torch.load(model_dir)['net'], strict=True)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    prediction_dir = None
    img_name_list = [path]
    # 1. dataload
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[], edge_name_list=[],
                                        transform=transforms.Compose([RescaleT(1536), ToTensor()]))
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        for i_test, data_test in enumerate(tqdm.tqdm(test_salobj_dataloader, ncols=60)):
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda(), requires_grad=False)
            else:
                inputs_test = Variable(inputs_test)
            dout = net(inputs_test)[0]
            # normalization
            pred = dout[:, 0, :, :]
            pred = normPRED(pred)
            res = save_output_cv2(
                img_name_list[i_test], pred, prediction_dir, data_test['original_shape'])
            del dout, pred, inputs_test, data_test
            res = Image.fromarray(res)
    del net
    logger.info("rmf seg end.....")
    return res
    

if __name__ == '__main__':
    
    from loguru import logger

    # --------- 1. model define ---------
    logger.info("beging.....")
    args = myParser()
    args.pretrained_path=r'/root/fengxiaoqin/RMFormer/save_models/pretrain/swin_base_patch4_window12_384_22k.pth'
    model_dir =r'/root/fengxiaoqin/RMFormer/save_models/Atemp/model_UH_final.pth'
    print("...load MyNet...")
    net = myNet1(args)
    net.load_state_dict(torch.load(model_dir)['net'], strict=True)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    import os
    for _ in os.listdir(r'/root/fengxiaoqin/RMFormer/koutu'):
        path='/root/fengxiaoqin/RMFormer/koutu'+'/'+_
        res = rmf_seg(path, net)
        out='/root/fengxiaoqin/RMFormer/out2'+'/'+_
        res.save(out)
    logger.info("ending.....")
