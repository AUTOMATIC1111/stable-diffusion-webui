import unittest
from PIL import Image
from trainx.text_inversion import interrogators, WaifuDiffusionInterrogator


class TestWDInterrogator(unittest.TestCase):

    def test_wd(self):
        image_path = '/Users/wangdongming/Downloads/gm2.png'
        caption_wd_interrogator_name = 'wd14-vit-v2'
        caption_wd_interrogator = interrogators[caption_wd_interrogator_name]
        _, tags = caption_wd_interrogator.interrogate(Image.open(image_path))
        processed_tags = WaifuDiffusionInterrogator.postprocess_tags(
            tags,
            threshold=0.35
        )

        self.assertTrue(processed_tags != "")  # add assertion here


if __name__ == '__main__':
    unittest.main()
