import logging

from annotator.lama.saicinpainting.training.modules.ffc import FFCResNetGenerator
from annotator.lama.saicinpainting.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator

def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    
    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')
