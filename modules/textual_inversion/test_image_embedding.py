import os

import numpy as np
import torch
from PIL import Image

from modules.textual_inversion.image_embedding import (
    extract_image_data_embed,
    embedding_from_b64,
    caption_image_overlay,
    insert_image_data_embed,
    lcg,
)


def test_image_embed_read():
    testEmbed = Image.open(
        os.path.join(os.path.dirname(__file__), "test_embedding.png")
    )

    data = extract_image_data_embed(testEmbed)
    assert data is not None

    data = embedding_from_b64(testEmbed.text["sd-ti-embedding"])
    assert data is not None


def test_image_embed_write():
    image = Image.new("RGBA", (512, 512), (255, 255, 200, 255))
    cap_image = caption_image_overlay(
        image, "title", "footerLeft", "footerMid", "footerRight"
    )

    test_embed = {
        "string_to_param": {"*": torch.from_numpy(np.random.random((2, 4096)))}
    }

    embedded_image = insert_image_data_embed(cap_image, test_embed)

    retrived_embed = extract_image_data_embed(embedded_image)

    assert str(retrived_embed) == str(test_embed)

    embedded_image2 = insert_image_data_embed(cap_image, retrived_embed)

    assert embedded_image == embedded_image2


def test_lcg():
    g = lcg()
    shared_random = np.array([next(g) for _ in range(100)]).astype(np.uint8).tolist()

    reference_random = [
        253, 242, 127, 44, 157, 27, 239, 133, 38, 79, 167, 4, 177,
        95, 130, 79, 78, 14, 52, 215, 220, 194, 126, 28, 240, 179,
        160, 153, 149, 50, 105, 14, 21, 218, 199, 18, 54, 198, 193,
        38, 128, 19, 53, 195, 124, 75, 205, 12, 6, 145, 0, 28,
        30, 148, 8, 45, 218, 171, 55, 249, 97, 166, 12, 35, 0,
        41, 221, 122, 215, 170, 31, 113, 186, 97, 119, 31, 23, 185,
        66, 140, 30, 41, 37, 63, 137, 109, 216, 55, 159, 145, 82,
        204, 86, 73, 222, 44, 198, 118, 240, 97,
    ]

    assert shared_random == reference_random
    assert 12731374 == np.array([next(g) for _ in range(100000)]).astype(np.uint8).sum()
