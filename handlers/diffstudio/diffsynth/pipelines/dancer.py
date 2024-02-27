import torch
from ..models import SDUNet, SDMotionModel
from ..models.sd_unet import PushBlock, PopBlock
from ..models.tiler import TileWorker
from ..controlnets import MultiControlNetManager


def lets_dance(
    unet: SDUNet,
    motion_modules: SDMotionModel = None,
    controlnet: MultiControlNetManager = None,
    sample = None,
    timestep = None,
    encoder_hidden_states = None,
    controlnet_frames = None,
    unet_batch_size = 1,
    controlnet_batch_size = 1,
    cross_frame_attention = False,
    tiled=False,
    tile_size=64,
    tile_stride=32,
    device = "cuda",
    vram_limit_level = 0,
):
    # 1. ControlNet
    #     This part will be repeated on overlapping frames if animatediff_batch_size > animatediff_stride.
    #     I leave it here because I intend to do something interesting on the ControlNets.
    controlnet_insert_block_id = 30
    if controlnet is not None and controlnet_frames is not None:
        res_stacks = []
        # process controlnet frames with batch
        for batch_id in range(0, sample.shape[0], controlnet_batch_size):
            batch_id_ = min(batch_id + controlnet_batch_size, sample.shape[0])
            res_stack = controlnet(
                sample[batch_id: batch_id_],
                timestep,
                encoder_hidden_states[batch_id: batch_id_],
                controlnet_frames[:, batch_id: batch_id_],
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
            if vram_limit_level >= 1:
                res_stack = [res.cpu() for res in res_stack]
            res_stacks.append(res_stack)
        # concat the residual
        additional_res_stack = []
        for i in range(len(res_stacks[0])):
            res = torch.concat([res_stack[i] for res_stack in res_stacks], dim=0)
            additional_res_stack.append(res)
    else:
        additional_res_stack = None

    # 2. time
    time_emb = unet.time_proj(timestep[None]).to(sample.dtype)
    time_emb = unet.time_embedding(time_emb)

    # 3. pre-process
    height, width = sample.shape[2], sample.shape[3]
    hidden_states = unet.conv_in(sample)
    text_emb = encoder_hidden_states
    res_stack = [hidden_states.cpu() if vram_limit_level>=1 else hidden_states]

    # 4. blocks
    for block_id, block in enumerate(unet.blocks):
        # 4.1 UNet
        if isinstance(block, PushBlock):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
            if vram_limit_level>=1:
                res_stack[-1] = res_stack[-1].cpu()
        elif isinstance(block, PopBlock):
            if vram_limit_level>=1:
                res_stack[-1] = res_stack[-1].to(device)
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
        else:
            hidden_states_input = hidden_states
            hidden_states_output = []
            for batch_id in range(0, sample.shape[0], unet_batch_size):
                batch_id_ = min(batch_id + unet_batch_size, sample.shape[0])
                if tiled:
                    _, _, inter_height, _ = hidden_states.shape
                    resize_scale = inter_height / height
                    hidden_states = TileWorker().tiled_forward(
                        lambda x: block(x, time_emb, text_emb[batch_id: batch_id_], res_stack)[0],
                        hidden_states_input[batch_id: batch_id_],
                        int(tile_size * resize_scale),
                        int(tile_stride * resize_scale),
                        tile_device=hidden_states.device,
                        tile_dtype=hidden_states.dtype
                    )
                else:
                    hidden_states, _, _, _ = block(
                        hidden_states_input[batch_id: batch_id_],
                        time_emb,
                        text_emb[batch_id: batch_id_],
                        res_stack,
                        cross_frame_attention=cross_frame_attention
                    )
                hidden_states_output.append(hidden_states)
            hidden_states = torch.concat(hidden_states_output, dim=0)
        # 4.2 AnimateDiff
        if motion_modules is not None:
            if block_id in motion_modules.call_block_id:
                motion_module_id = motion_modules.call_block_id[block_id]
                hidden_states, time_emb, text_emb, res_stack = motion_modules.motion_modules[motion_module_id](
                    hidden_states, time_emb, text_emb, res_stack,
                    batch_size=1
                )
        # 4.3 ControlNet
        if block_id == controlnet_insert_block_id and additional_res_stack is not None:
            hidden_states += additional_res_stack.pop().to(device)
            if vram_limit_level>=1:
                res_stack = [(res.to(device) + additional_res.to(device)).cpu() for res, additional_res in zip(res_stack, additional_res_stack)]
            else:
                res_stack = [res + additional_res for res, additional_res in zip(res_stack, additional_res_stack)]
    
    # 5. output
    hidden_states = unet.conv_norm_out(hidden_states)
    hidden_states = unet.conv_act(hidden_states)
    hidden_states = unet.conv_out(hidden_states)

    return hidden_states
