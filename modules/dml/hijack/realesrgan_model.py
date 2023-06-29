import math
import torch

from realesrgan import RealESRGANer


# DML Solution: Some of contents of output tensor turn to 0 after Extended Slices. Move it to cpu.
def tile_process(self):
    batch, channel, height, width = self.img.shape
    output_height = height * self.scale
    output_width = width * self.scale
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    self.output = self.img.new_zeros(output_shape)
    tiles_x = math.ceil(width / self.tile_size)
    tiles_y = math.ceil(height / self.tile_size)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * self.tile_size
            ofs_y = y * self.tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + self.tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + self.tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - self.tile_pad, 0)
            input_end_x_pad = min(input_end_x + self.tile_pad, width)
            input_start_y_pad = max(input_start_y - self.tile_pad, 0)
            input_end_y_pad = min(input_end_y + self.tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = self.img[0:self.img.shape[0], 0:self.img.shape[1], input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            try:
                with torch.no_grad():
                    output_tile = self.model(input_tile)
            except RuntimeError as error:
                print('Error', error)
            print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

            # output tile area on total image
            output_start_x = input_start_x * self.scale
            output_end_x = input_end_x * self.scale
            output_start_y = input_start_y * self.scale
            output_end_y = input_end_y * self.scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

            self.output = self.output.cpu()
            # put tile into output image
            self.output[0:self.output.shape[0], 0:self.output.shape[1], output_start_y:output_end_y, output_start_x:output_end_x] = output_tile.cpu()[0:output_tile.shape[0], 0:output_tile.shape[1], output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
            self.output = self.output.to(output_tile.device)
RealESRGANer.tile_process = tile_process
