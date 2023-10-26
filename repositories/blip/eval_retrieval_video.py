'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval import blip_retrieval
import utils
from data.video_dataset import VideoDataset


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = tokenizer.additional_special_tokens_ids[0]
    
    video_feats = []
    video_embeds = []
    for video, video_id in data_loader: 

        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H)
        video = video.to(device,non_blocking=True) 
        video_feat = model.visual_encoder(video)        
        video_embed = model.vision_proj(video_feat[:,0,:])   
        video_embed = video_embed.view(B,N,-1).mean(dim=1)
        video_embed = F.normalize(video_embed,dim=-1)  
       
        video_feat = video_feat.view(B,-1,video_feat.shape[-1])
        video_feats.append(video_feat.cpu())
        video_embeds.append(video_embed)
     
    video_feats = torch.cat(video_feats,dim=0)
    video_embeds = torch.cat(video_embeds,dim=0)
    
    sims_matrix = video_embeds @ text_embeds.t()
    score_matrix_v2t = torch.full((len(texts),len(texts)),-100.0).to(device) 
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        
        encoder_output = video_feats[start+i].repeat(config['k_test'],1,1).to(device,non_blocking=True) 
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True) 
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_v2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2v = torch.full((len(texts),len(texts)),-100.0).to(device) 
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = video_feats[topk_idx].to(device,non_blocking=True) 
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True) 
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2v[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_v2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2v, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_v2t.cpu().numpy(), score_matrix_t2v.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_v2t, scores_t2v, txt2vmg, vid2txt):
    
    #Video->Text 
    ranks = np.zeros(scores_v2t.shape[0])
    for index,score in enumerate(scores_v2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == vid2txt[index])[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Video 
    ranks = np.zeros(scores_t2v.shape[0])
    
    for index,score in enumerate(scores_t2v):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2vmg[index])[0][0]
    
    mdR = np.median(ranks+1)
        
    # Compute metrics
    vr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    vr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    vr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    vr_mean = (vr1 + vr5 + vr10) / 3
    r_mean = (tr_mean + vr_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'vid_r1': vr1,
                    'vid_r5': vr5,
                    'vid_r10': vr10,
                    'vid_r_mean': vr_mean,
                    'vid_mdR': mdR,
                    'r_mean': r_mean}
    return eval_result




def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    test_dataset = VideoDataset(config['video_root'],config['ann_root'],num_frm=config['num_frm_test'],
                                max_img_size=config['image_size'], frm_sampling_strategy='uniform')

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )  

    #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'])
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    score_v2t, score_t2v, = evaluation(model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config)

    if utils.is_main_process():  

        test_result = itm_eval(score_v2t, score_t2v, test_loader.dataset.txt2video, test_loader.dataset.video2txt)  
        print(test_result)

        log_stats = {**{f'{k}': v for k, v in test_result.items()},}
        with open(os.path.join(args.output_dir, "test_result.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_msrvtt.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_msrvtt')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)