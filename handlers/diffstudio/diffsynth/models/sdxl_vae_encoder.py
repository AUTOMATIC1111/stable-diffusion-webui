from .sd_vae_encoder import SDVAEEncoderStateDictConverter, SDVAEEncoder


class SDXLVAEEncoder(SDVAEEncoder):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.13025
    
    def state_dict_converter(self):
        return SDXLVAEEncoderStateDictConverter()


class SDXLVAEEncoderStateDictConverter(SDVAEEncoderStateDictConverter):
    def __init__(self):
        super().__init__()
