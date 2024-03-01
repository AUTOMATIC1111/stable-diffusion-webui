from .sd_vae_decoder import SDVAEDecoder, SDVAEDecoderStateDictConverter


class SDXLVAEDecoder(SDVAEDecoder):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.13025

    def state_dict_converter(self):
        return SDXLVAEDecoderStateDictConverter()
    

class SDXLVAEDecoderStateDictConverter(SDVAEDecoderStateDictConverter):
    def __init__(self):
        super().__init__()
