import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.rife.model_ifnet import IFNet
from modules.rife.loss import EPE, SOBEL
from modules import devices


class RifeModel:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        self.version = 3.9
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(devices.device)
        self.flownet.to(devices.dtype)

    def load_model(self, model_file, rank=0):
        def convert(param):
            if rank == -1:
                return { k.replace("module.", ""): v for k, v in param.items() if "module." in k }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(model_file)), False)
            else:
                self.flownet.load_state_dict(convert(torch.load(model_file, map_location='cpu')), False)

    def save_model(self, model_file, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), model_file)

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        _flow, _mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None): # pylint: disable=unused-argument
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        # img0 = imgs[:, :3]
        # img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [8, 4, 2, 1]
        flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[3] - gt).abs().mean()
        loss_smooth = self.sobel(flow[3], flow[3]*0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        # else:
        #    flow_teacher = flow[2]
        return merged[3], {
            'mask': mask,
            'flow': flow[3][:, :2],
            'loss_l1': loss_l1,
            'loss_smooth': loss_smooth,
        }
