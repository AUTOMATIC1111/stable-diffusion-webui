import torch
from einops import rearrange, repeat


class Tiler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def mask(self, height, width, line_width):
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / line_width).clip(0, 1)
        return mask

    def forward(self, forward_fn, x, tile_size, tile_stride, batch_size=1, inter_device="cpu", inter_dtype=torch.float32):
        # Prepare
        device = x.device
        torch_dtype = x.dtype

        # tile
        b, c_in, h_in, w_in = x.shape
        x = x.to(device=inter_device, dtype=inter_dtype)
        fold_params = {
            "kernel_size": (tile_size, tile_size),
            "stride": (tile_stride, tile_stride)
        }
        unfold_operator = torch.nn.Unfold(**fold_params)
        x = unfold_operator(x)
        x = x.view((b, c_in, tile_size, tile_size, -1))

        # inference
        x_out_stack = []
        for tile_id in range(0, x.shape[-1], batch_size):

            # process input
            next_tile_id = min(tile_id + batch_size, x.shape[-1])
            x_in = x[:, :, :, :, tile_id: next_tile_id]
            x_in = x_in.to(device=device, dtype=torch_dtype)
            x_in = x_in.permute(4, 0, 1, 2, 3)
            x_in = x_in.view((x_in.shape[0]*x_in.shape[1], x_in.shape[2], x_in.shape[3], x_in.shape[4]))

            # process output
            x_out = forward_fn(x_in)
            x_out = x_out.view((next_tile_id - tile_id, b, x_out.shape[1], x_out.shape[2], x_out.shape[3]))
            x_out = x_out.permute(1, 2, 3, 4, 0)
            x_out = x_out.to(device=inter_device, dtype=inter_dtype)
            x_out_stack.append(x_out)

        x = torch.concat(x_out_stack, dim=-1)

        # untile
        in2out_scale = x.shape[2] / tile_size
        h_out, w_out = int(h_in * in2out_scale), int(w_in * in2out_scale)

        mask = self.mask(int(tile_size * in2out_scale), int(tile_size * in2out_scale), int(tile_stride * in2out_scale * 0.5))
        mask = mask.to(device=inter_device, dtype=inter_dtype)
        mask = mask.reshape((1, 1, mask.shape[0], mask.shape[1], 1))
        x = x * mask

        fold_params = {
            "kernel_size": (int(tile_size * in2out_scale), int(tile_size * in2out_scale)),
            "stride": (int(tile_stride * in2out_scale), int(tile_stride * in2out_scale))
        }
        fold_operator = torch.nn.Fold(output_size=(h_out, w_out), **fold_params)
        divisor = fold_operator(mask.repeat(1, 1, 1, 1, x.shape[-1]).view(b, -1, x.shape[-1]))

        x = x.view((b, -1, x.shape[-1]))
        x = fold_operator(x) / divisor
        x = x.to(device=device, dtype=torch_dtype)

        return x


class TileWorker:
    def __init__(self):
        pass


    def mask(self, height, width, border_width):
        # Create a mask with shape (height, width).
        # The centre area is filled with 1, and the border line is filled with values in range (0, 1].
        x = torch.arange(height).repeat(width, 1).T
        y = torch.arange(width).repeat(height, 1)
        mask = torch.stack([x + 1, height - x, y + 1, width - y]).min(dim=0).values
        mask = (mask / border_width).clip(0, 1)
        return mask


    def tile(self, model_input, tile_size, tile_stride, tile_device, tile_dtype):
        # Convert a tensor (b, c, h, w) to (b, c, tile_size, tile_size, tile_num)
        batch_size, channel, _, _ = model_input.shape
        model_input = model_input.to(device=tile_device, dtype=tile_dtype)
        unfold_operator = torch.nn.Unfold(
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride)
        )
        model_input = unfold_operator(model_input)
        model_input = model_input.view((batch_size, channel, tile_size, tile_size, -1))

        return model_input


    def tiled_inference(self, forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype):
        # Call y=forward_fn(x) for each tile
        tile_num = model_input.shape[-1]
        model_output_stack = []

        for tile_id in range(0, tile_num, tile_batch_size):

            # process input
            tile_id_ = min(tile_id + tile_batch_size, tile_num)
            x = model_input[:, :, :, :, tile_id: tile_id_]
            x = x.to(device=inference_device, dtype=inference_dtype)
            x = rearrange(x, "b c h w n -> (n b) c h w")

            # process output
            y = forward_fn(x)
            y = rearrange(y, "(n b) c h w -> b c h w n", n=tile_id_-tile_id)
            y = y.to(device=tile_device, dtype=tile_dtype)
            model_output_stack.append(y)

        model_output = torch.concat(model_output_stack, dim=-1)
        return model_output


    def io_scale(self, model_output, tile_size):
        # Determine the size modification happend in forward_fn
        # We only consider the same scale on height and width.
        io_scale = model_output.shape[2] / tile_size
        return io_scale
    

    def untile(self, model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype):
        # The reversed function of tile
        mask = self.mask(tile_size, tile_size, border_width)
        mask = mask.to(device=tile_device, dtype=tile_dtype)
        mask = rearrange(mask, "h w -> 1 1 h w 1")
        model_output = model_output * mask

        fold_operator = torch.nn.Fold(
            output_size=(height, width),
            kernel_size=(tile_size, tile_size),
            stride=(tile_stride, tile_stride)
        )
        mask = repeat(mask[0, 0, :, :, 0], "h w -> 1 (h w) n", n=model_output.shape[-1])
        model_output = rearrange(model_output, "b c h w n -> b (c h w) n")
        model_output = fold_operator(model_output) / fold_operator(mask)

        return model_output


    def tiled_forward(self, forward_fn, model_input, tile_size, tile_stride, tile_batch_size=1, tile_device="cpu", tile_dtype=torch.float32, border_width=None):
        # Prepare
        inference_device, inference_dtype = model_input.device, model_input.dtype
        height, width = model_input.shape[2], model_input.shape[3]
        border_width = int(tile_stride*0.5) if border_width is None else border_width

        # tile
        model_input = self.tile(model_input, tile_size, tile_stride, tile_device, tile_dtype)

        # inference
        model_output = self.tiled_inference(forward_fn, model_input, tile_batch_size, inference_device, inference_dtype, tile_device, tile_dtype)

        # resize
        io_scale = self.io_scale(model_output, tile_size)
        height, width = int(height*io_scale), int(width*io_scale)
        tile_size, tile_stride = int(tile_size*io_scale), int(tile_stride*io_scale)
        border_width = int(border_width*io_scale)

        # untile
        model_output = self.untile(model_output, height, width, tile_size, tile_stride, border_width, tile_device, tile_dtype)
        
        # Done!
        model_output = model_output.to(device=inference_device, dtype=inference_dtype)
        return model_output