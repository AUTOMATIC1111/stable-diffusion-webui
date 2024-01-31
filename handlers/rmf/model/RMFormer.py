import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper.helper_blocks import *
from .helper.custom_trans_v1 import Swin, PatchEmbed_My, Mlp
import cv2
import os

def save_output_cv2(pred_in,num_name):
    pred = pred_in[:,0,:,:].detach()

    pred_ma = torch.max(pred)
    pred_mi = torch.min(pred)

    pred = (pred-pred_mi)/(pred_ma-pred_mi)

    image_name = 'mytestfeature'+str(num_name)+'.png'
    d_dir = 'test_data/result/'
    origin_shape = (pred.shape[-2],pred.shape[-1])

    predict = pred.cpu().numpy().transpose((1,2,0))
    # im = cv2.cvtColor((predict*255),cv2.COLOR_RGB2BGR)
    im = predict*255
    imo = cv2.resize(im,(int(origin_shape[1]),int(origin_shape[0])))

    aaa = image_name.split("/")[-1]
    imidx = os.path.splitext(aaa)[0]

    # print(d_dir+imidx+'.png')

    cv2.imwrite(d_dir+imidx+'.png',imo)

def save_feature_cv2(pred_in,num_name):
    C = pred_in.shape[1]
    pred_list = []
    for i in range(C):
        pred = pred_in[:,i,:,:].detach()
        pred_ma = torch.max(pred)
        pred_mi = torch.min(pred)

        pred = (pred-pred_mi)/(pred_ma-pred_mi)
        pred_list.append(pred)
    image_name = 'mytestfeature'+str(num_name)+'.png'
    d_dir = 'test_data/result/'
    ori_shape = pred_in.shape[-1]
    origin_shape = (ori_shape,ori_shape)

    img_out = np.zeros(origin_shape)
    final = np.zeros((ori_shape,ori_shape*9))
    i = 1
    for p in pred_list:
        predict = p.cpu().numpy().transpose((1,2,0))
        # im = cv2.cvtColor((predict*255),cv2.COLOR_RGB2BGR)
        im = predict*255
        imo = cv2.resize(im,(int(origin_shape[1]),int(origin_shape[0])))
        img_out = np.hstack((img_out,imo))
        if (i%8) == 0:
            final = np.vstack((final,img_out))
            img_out = np.zeros(origin_shape)
        i += 1
    aaa = image_name.split("/")[-1]
    imidx = os.path.splitext(aaa)[0]
    cv2.imwrite(d_dir+imidx+'.png',final)


class RMF(nn.Module):
    def __init__(self, config):
        super(RMF,self).__init__()
        self.lr_res = 384

        # CPS
        self.lr_branch = Swin(img_size=self.lr_res,
                                patch_size=config.patch_size,
                                in_chans=config.in_chans,
                                num_classes=config.num_classes,
                                embed_dim=config.embed_dim,
                                depths=config.depth,
                                num_heads=config.num_heads,
                                window_size=config.window_size,
                                mlp_ratio=config.mlp_ratio,
                                qkv_bias=config.qkv_bias,
                                qk_scale=config.qk_scale,
                                drop_rate=config.drop_rate,
                                drop_path_rate=config.drop_path_rate,
                                ape=config.ape,
                                patch_norm=config.patch_norm,
                                use_checkpoint=config.use_checkpoint,
                                conv_channel=32)
        if config.use_pretrain:
            self.load_from(config)
        self.predictor = MyDecoder(hrimg_size=384,tf_dim=128)

        # PR
        self.pr = PixelRefiner(cur_dim=16)

        # RRS
        self.rrs1 = RRS(self.lr_branch,self.predictor, self.pr, stage_res=768)
        self.rrs2 = RRS(self.lr_branch,self.predictor, self.pr, stage_res=1536)
        



    def forward(self,x):
        lr_input = F.interpolate(x, self.lr_res, mode='bilinear', align_corners=False)
        lr_emb_list = self.lr_branch(lr_input)


        out_pred_list = []
        out_edge_list = []
        out_attn_list = []


        pred_c,lr_f, pred_e, proto = self.predictor(lr_emb_list, '0')
        out_edge_list.append(pred_e)

        pred_mid, edge_mid, attn_list = self.rrs1(pred_c, x)
        out_pred_list += pred_mid
        out_edge_list += edge_mid
        out_attn_list += attn_list

        pred_mid, edge_mid, attn_list = self.rrs2(pred_mid[-1], x)
        out_pred_list += pred_mid
        out_edge_list += edge_mid
        out_attn_list += attn_list

        pred_final = out_pred_list[-1]
        if not self.training:
            pred_final = torch.sigmoid(pred_final)

        return pred_final,pred_c, out_pred_list, out_edge_list, out_attn_list

    def load_from(self, config):
        pretrained_path = config.pretrained_path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.lr_branch.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.lr_branch.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.lr_branch.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")



class decode_MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

        self.initialize()
        

    def forward(self, x):
        B_,N_,_ = x.shape
        res = int(np.sqrt(N_))
        x = self.proj(x)

        x = x.view(B_,res,res,-1).permute(0,3,1,2)
        return x

    def initialize(self):
        weight_init_my(self)

class MyDecoder(nn.Module):
    def __init__(self ,hrimg_size = 1024, tf_dim = 128):
        super(MyDecoder, self).__init__()
        self.tf_dim2 = tf_dim

        # follow SegFormer
        self.unemb2_1 = decode_MLP(self.tf_dim2*1,self.tf_dim2*1)

        self.unemb2 = nn.Sequential(
            decode_MLP(self.tf_dim2*1,self.tf_dim2*1),
            decode_MLP(self.tf_dim2*2,self.tf_dim2*1),
            decode_MLP(self.tf_dim2*4,self.tf_dim2*1),
        )

        self.unemb_norm = nn.Sequential(
            nn.LayerNorm(self.tf_dim2*1),
            nn.LayerNorm(self.tf_dim2*2),
            nn.LayerNorm(self.tf_dim2*4),
        )


        self.unemb_final1 = decode_MLP(self.tf_dim2*8,self.tf_dim2*1)
        self.unemb_final2 = decode_MLP(self.tf_dim2*8,self.tf_dim2*1)
        self.unemb_final3 = decode_MLP(self.tf_dim2*8,self.tf_dim2*1)

        self.norm_final2 = nn.LayerNorm(self.tf_dim2*8)
        self.norm_final3 = nn.LayerNorm(self.tf_dim2*8)

        self.pred_conv_c1 = ConvBlock(self.tf_dim2, self.tf_dim2//8, 1,0)
        self.pred_conv_c1_final = nn.Conv2d(self.tf_dim2//8,1,kernel_size=1,bias=False)

        self.pred_conv_c2 = ConvBlock(self.tf_dim2, self.tf_dim2//8, 1,0)
        self.pred_conv_c2_final = nn.Conv2d(self.tf_dim2//8,1,kernel_size=1,bias=False)
        self.pred_conv_c3 = ConvBlock(self.tf_dim2, self.tf_dim2//8, 1,0)
        self.pred_conv_c3_final = nn.Conv2d(self.tf_dim2//8,1,kernel_size=1,bias=False)

        self.pred_conv_e1_final = nn.Conv2d(self.tf_dim2//8,1,kernel_size=1,bias=False)
        self.pred_conv_e2_final = nn.Conv2d(self.tf_dim2//8,1,kernel_size=1,bias=False)
        self.pred_conv_e3_final = nn.Conv2d(self.tf_dim2//8,1,kernel_size=1,bias=False)

        self.conv_shrink_1 = ConvBlock(self.tf_dim2*4, self.tf_dim2*1,1,0)
        self.conv_shrink_2 = ConvBlock(self.tf_dim2*4, self.tf_dim2*1,1,0)
        self.conv_shrink_3 = ConvBlock(self.tf_dim2*4, self.tf_dim2*1,1,0)

        self.initialize()



    def forward(self, x2, stage='0'):
        lr_x_list = []
        lr_outlist = x2[1]
        lr_1 = self.unemb2_1(lr_outlist[0])
        lr_x_list.append(lr_1)

        res1of4 = lr_1.shape[-1]
        num_stage = len(lr_outlist)-1


        for i in range(num_stage):
            if i==0:
                continue
            lr_x = self.unemb2[i](self.unemb_norm[i](lr_outlist[i]))
            lr_x = F.interpolate(lr_x,res1of4,mode='bilinear',align_corners=False)
            lr_x_list.append(lr_x)

        if stage == '0':
            lr_x = self.unemb_final1(x2[0])
        elif stage == '1':
            lr_x = self.unemb_final2(self.norm_final2(x2[0]))
        else:
            lr_x = self.unemb_final3(self.norm_final3(x2[0]))

        lr_x = F.interpolate(lr_x,res1of4,mode='bilinear',align_corners=False)
        lr_x_list.append(lr_x)
        lr_out = torch.hstack(lr_x_list)

        if stage == '0':
            lr_out = self.conv_shrink_1(lr_out)
            lr_out_proto = self.pred_conv_c1(lr_out)
            lr_out = F.interpolate(lr_out_proto,scale_factor=4,mode='bilinear',align_corners=False)
            pred_c = self.pred_conv_c1_final(lr_out)
            pred_e = self.pred_conv_e1_final(lr_out)

        elif stage == '1':
            lr_out = self.conv_shrink_2(lr_out)
            lr_out_proto = self.pred_conv_c2(lr_out)
            lr_out = F.interpolate(lr_out_proto,scale_factor=4,mode='bilinear',align_corners=False)
            pred_c = self.pred_conv_c2_final(lr_out)
            pred_e = self.pred_conv_e2_final(lr_out)
        else:
            lr_out = self.conv_shrink_3(lr_out)
            lr_out_proto = self.pred_conv_c3(lr_out)
            lr_out = F.interpolate(lr_out_proto,scale_factor=4,mode='bilinear',align_corners=False)
            pred_c = self.pred_conv_c3_final(lr_out)
            pred_e = self.pred_conv_e3_final(lr_out)

        return pred_c, lr_out, pred_e, lr_out_proto


    def initialize(self):
        weight_init_my(self)


class RRS(nn.Module):
    def __init__(self ,lrswinnet,decoder, prrefine, stage_res = 1024):
        super(RRS, self).__init__()

        self.stage_res = stage_res

        if stage_res == 768:
            self.cps_num = '1'
        elif stage_res == 1536:
            self.cps_num = '2'

        self.down_num = int((self.stage_res / 384)//2)
        self.patchsize2 = 4
        self.tf_dim2 = 128
        self.lr_res = 384
        self.emb_dim = 128

        self.pr_stage = [768, 1536]



        self.patch_embed1 = PatchEmbed_My(
            img_size=384, patch_size=4, in_chans=16+3, embed_dim=self.emb_dim,
            norm_layer=nn.LayerNorm)

        self.in_conv = ConvBlock(3, 16)

        self.stageconv = nn.ModuleList(
            ConvBlock(16+3, 16) for _ in range(self.down_num)
        )

        self.mid_conv = ConvBlock(16+2, 16)

        self.deconv = nn.ModuleList(
            ConvBlock(16+3, 16) for _ in range(self.down_num)
        )     


        self.sideout = nn.ModuleList(
            nn.Conv2d(16,1,kernel_size=1,bias=False) for _ in range(self.down_num+1)
        )
        self.sideout_e = nn.ModuleList(
            nn.Conv2d(16,1,kernel_size=1,bias=False) for _ in range(self.down_num+1)
        )

        self.seghead = decoder

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.initialize()

        self.swinlayers = lrswinnet
        self.refiner = prrefine




    def forward(self, pred_c, img):
        pred_in = pred_c
        pred_in = torch.sigmoid(pred_in)

        # IGE =======================
        img_in = F.interpolate(img,self.stage_res,mode='bilinear',align_corners=False)
        x_in = img_in
        x_down_1 = []
        en_feats = []

        x_in = self.in_conv(x_in)
        en_feats.append(x_in)

        for convlayer in self.stageconv:
            x_in = self.down(x_in)
            img_in_t = F.interpolate(img,x_in.shape[-1],mode='bilinear',align_corners=False)
            x_in = torch.cat([x_in,img_in_t], 1)
            x_in = convlayer(x_in)
            en_feats.append(x_in)


        # CPS ===========================
        x_in = torch.cat([x_in, img_in_t], 1)
        tempx = self.patch_embed1(x_in)

        for i_layer in range(len(self.swinlayers.layers)):

            tempx,x_ori = self.swinlayers.layers[i_layer](tempx)
            x_down_1.append(x_ori)

        pred_mid, f_up, edge_mid, proto = self.seghead([tempx, x_down_1], self.cps_num)



        # DGD ===========================
        pred_list = []
        edge_list = []
        attn_list = []


        pred_list.append(pred_mid)
        edge_list.append(edge_mid)

        for j in range(self.down_num+1):
            cur_res = f_up.shape[-1]
            pred_in_t = F.interpolate(pred_in,cur_res,mode='bilinear',align_corners=False)
            edge_in_t = F.interpolate(torch.sigmoid(edge_list[-1]),cur_res,mode='bilinear',align_corners=False)

            if cur_res in self.pr_stage:
                pred_refine, attnout1 = self.refiner(en_feats[-1-j], en_feats[-1], f_up, pred_in_t,edge_in_t, proto)
                attn_list.append(attnout1)
                f_up = self.deconv[-1-j+1](torch.cat([f_up+en_feats[-1-j], pred_in_t,edge_in_t,pred_refine], 1))
            else:
                f_up = self.mid_conv(torch.cat([f_up+en_feats[-1-j], pred_in_t,edge_in_t], 1))
            pred_list.append(self.sideout[-1-j](f_up))
            edge_list.append(self.sideout_e[-1-j](f_up))
            f_up = self.up(f_up)

        return pred_list, edge_list, attn_list

    def initialize(self):
        weight_init_my(self)


class PixelRefiner(nn.Module):
    def __init__(self ,cur_dim=32, conv_dim=16):
        super(PixelRefiner, self).__init__()
        self.cur_dim = cur_dim
        self.conv_dim = conv_dim

        self.cross_enhance = My_Attn_Cross(self.conv_dim, self.cur_dim)

        self.pixel_out = Mlp(self.conv_dim, self.conv_dim, 1)

        self.down_conv = nn.Sequential(
            nn.Conv2d(self.cur_dim,self.cur_dim,kernel_size=2,stride=2, bias=False),
            nn.BatchNorm2d(self.cur_dim),
            nn.Conv2d(self.cur_dim,self.cur_dim,kernel_size=2,stride=2, bias=False),
            nn.BatchNorm2d(self.cur_dim),
            nn.Conv2d(self.cur_dim,self.cur_dim,kernel_size=2,stride=2, bias=False),
            nn.BatchNorm2d(self.cur_dim),
        )
        self.proto_cal = My_Attn(self.cur_dim)
        self.act = nn.GELU()

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(self.cur_dim,self.cur_dim,kernel_size=2,stride=2, bias=False),
            nn.BatchNorm2d(self.cur_dim),
            nn.Conv2d(self.cur_dim,self.cur_dim,kernel_size=4,stride=4, bias=False),
            nn.BatchNorm2d(self.cur_dim),
            nn.Conv2d(self.cur_dim,self.cur_dim,kernel_size=4,stride=4, bias=False),
            nn.BatchNorm2d(self.cur_dim),
        )
        self.proto_cal1 = nn.Sequential(nn.GELU(), My_Attn(self.cur_dim))

        self.initialize()

    def forward(self, conv_hr, conv_lr, de, pred_map, edge_map, sam_proto):
        res_cur = sam_proto.shape[-1]
        des_red = res_cur//8

        out_de = []
        attn_out_list = []
        pred_de = []


        cur_hr_conv = conv_hr
        cur_de = de

        sam_proto_residul = F.interpolate(sam_proto, des_red, mode='bilinear')
        conv_lr_residul = F.interpolate(conv_lr, des_red, mode='bilinear')

        conv_sam_flatten = self.proto_cal(self.act(self.down_conv(sam_proto) + sam_proto_residul).flatten(-2,-1).permute(0,2,1))
        conv_lr_flatten = self.proto_cal1(self.act(self.down_conv1(conv_lr) + conv_lr_residul).flatten(-2,-1).permute(0,2,1))


        if len(cur_hr_conv.shape) == 4:
            cur_hr_conv = cur_hr_conv.flatten(-2,-1).permute(0,2,1)
            cur_de = cur_de.flatten(-2,-1).permute(0,2,1)
            pred_map = pred_map.flatten(-2,-1).permute(0,2,1)
            edge_map = edge_map.flatten(-2,-1).permute(0,2,1)

        B, N, C = cur_hr_conv.shape
        cur_size = int(np.sqrt(N))

        if self.training:
            s_factor = 200
        else:
            s_factor = 20

        select_conv,select_de, idx_group = self.patch_gen_select(edge_map, cur_hr_conv, cur_de, select_factor=s_factor)

        conv_en_select = self.cross_enhance(select_conv, conv_lr_flatten, conv_sam_flatten, shortcut=select_de)

        attn_out = self.pixel_out(conv_en_select)

        attn_out_list.append([attn_out, idx_group[0]])

        attn_p = torch.sigmoid(attn_out)

        pred_en = self.patch_reverse(conv_en_select, attn_p,cur_de, pred_map, idx_group)
        pred_de = pred_en.view(B, cur_size, cur_size, -1).permute(0,3,1,2)

        return pred_de, attn_out_list

    def initialize(self):
        weight_init_my(self)

    def patch_gen_select(self, judge_map, emb_ori, de_ori, select_factor = 40):
        B_t, N_t, C_t = emb_ori.shape
        C_t2 = de_ori.shape[-1]

        if self.training:
            judge_map_thre = torch.where(judge_map>0.1, 1, -1)
            rand_mask = torch.rand_like(judge_map).cuda()
            judge_map_thre = judge_map_thre * rand_mask
        else:
            judge_map_thre = torch.where(judge_map>0.1, 1, -1)
            rand_mask = torch.rand_like(judge_map).cuda()
            judge_map_thre = judge_map_thre * rand_mask

        _, topkindex_edge = judge_map_thre.topk(N_t//select_factor,dim=1, largest = True)

        topkindex_edge1 = topkindex_edge.expand(-1,-1, C_t)
        topkindex_edge2 = topkindex_edge.expand(-1,-1, C_t2)


        emb_topk = emb_ori.gather(1,topkindex_edge1)
        de_topk = de_ori.gather(1,topkindex_edge2)
        # for small object, find selected 0 position
        # zero_mask = torch.zeros_like(edge_tf_topk)

        return emb_topk,de_topk, [topkindex_edge, topkindex_edge1, topkindex_edge2]

    def patch_reverse(self, emb_select, attn_p, emb_ori, pred_map, idx_g):
        B_t, N_t, C_t = emb_ori.shape

        pred_out_scatter = pred_map.clone()
        pred_out_scatter = pred_out_scatter.scatter(1, idx_g[0], attn_p)


        return pred_out_scatter


class My_Attn_Cross(nn.Module):
    def __init__(self, dim, vdim, num_heads=1, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim,dim*4,dim, act_layer=nn.GELU, drop=0.)
        self.attn = nn.MultiheadAttention(dim,
                                          num_heads=1,
                                          dropout=0.1,
                                          kdim=vdim,
                                          vdim=vdim,
                                          batch_first=True)
        self.initialize()

    def forward(self, qin, kin, vin, shortcut=None):
        x = self.attn(query = qin, key = kin, value = vin)[0]

        # If nn.MultiheadAttention has no batch_first keyword, try to upgrade PyTorch>= 1.9.0 or use codes below:

        # qin = qin.transpose(1,0)
        # kin = kin.transpose(1,0)
        # vin = vin.transpose(1,0)
        # x = self.attn(query = qin, key = kin, value = vin)[0]
        # x = x.transpose(1,0)

        if shortcut is not None:
            pre_conv_hr = shortcut + self.proj_drop(self.norm1(self.mlp(x)))
        else:
            pre_conv_hr = self.proj_drop(self.norm1(self.mlp(x)))
        return pre_conv_hr

    def initialize(self):
        weight_init_my(self)

class My_Attn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim,dim*4,dim, act_layer=nn.GELU, drop=0.)

        self.initialize()

    def forward(self, x_in):
        B, N, C = x_in.shape
        qkv = self.qkv(x_in).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        pre_conv_hr = x_in + self.proj_drop(self.norm1(x))
        pre_conv_hr = pre_conv_hr + self.proj_drop(self.norm2(self.mlp(pre_conv_hr)))
        return pre_conv_hr

    def initialize(self):
        weight_init_my(self)
