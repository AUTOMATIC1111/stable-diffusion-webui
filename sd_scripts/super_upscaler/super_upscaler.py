import os
import sd_scripts.super_upscaler.esrgan_upscaler as esrgan_upscaler
from sd_scripts.super_upscaler.realesrgan_upscaler import RealESRGANUpscaler
from PIL import Image
# from modules.paths_internal import models_path

"""
图像放大：
a)	三次元：R-ESRGAN + ULTRASHARP（0.3权重），或者两个放大器对调位置
b)	二次元：R-ESRGAN ANIME + ULTRASHARP（0.3权重），或者两个放大器对调位置
"""

# UltraSharp_path = './4x-UltraSharp.pth'


def upscaler_inner(image, upscale_by, upscaler, upscaler2_path, upscaler_2_visibility, swap=False, models_path=""):
    # upscaler2 = "ULTRASHARP"
    # 第一个放大器
    first_upscaled_image = upscaler.upscale(image, upscale_by)
    # 第二个放大器
    second_upscaled_image = esrgan_upscaler.upscale(image, upscale_by, upscaler2_path)

    if not swap:
        upscaled_image = Image.blend(first_upscaled_image, second_upscaled_image, upscaler_2_visibility)
    else:
        upscaled_image = Image.blend(second_upscaled_image, first_upscaled_image, upscaler_2_visibility)

    return upscaled_image


def upscaler(init_image_ori, upscale_by=4, style_type=0, upscaler_2_visibility=0.3, swap=True,models_path=""):
    # image_path: 图片路径
    # upscale_by: 图片放大倍数
    # style_type:  0-二次元，1-三次元

    upscaler2_path = os.path.join(models_path,'UltraSharp/4x-UltraSharp.pth')
    
    image_path=init_image_ori
    if not isinstance(image_path,Image.Image):
        image = Image.open(image_path)
    else:
        image=image_path
    
    if image.mode != "RGB":
        # 将图像转换为RGB模式
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image
    img = upscaler_inner(rgb_image, upscale_by, RealESRGANUpscaler(style_type,models_path), upscaler2_path, upscaler_2_visibility, swap)

    return img

def test():
    
    upscale_by=4
    # for i in range(10):
    #     image_path = '/data/qll/sd-super-functions/sanciyuan.jpg'
    #     up_visi=i*0.1+0.1
    #     img = upscaler(image_path, upscale_by, style_type=1, upscaler_2_visibility=up_visi, swap=False)
    #     img.save(f'./sanciyuan-{up_visi}-ersgan-ultrasharp.jpg')
    #     img = upscaler(image_path, upscale_by, style_type=1, upscaler_2_visibility=up_visi, swap=True)
    #     img.save(f'./sanciyuan-{up_visi}-ultrasharp-ersgan.jpg')
    models_path = ""
    for i in range(10):
        image_path = '/data/qll/sd-super-functions/erciyuan.jpg'
        up_visi=i*0.1+0.1
        img = upscaler(image_path, upscale_by, style_type=0, upscaler_2_visibility=up_visi, swap=False,models_path=models_path)
        img.save(f'./erciyuan-{up_visi}-ersgan-ultrasharp.jpg')
        img = upscaler(image_path, upscale_by, style_type=0, upscaler_2_visibility=up_visi, swap=True,models_path=models_path)
        img.save(f'./erciyuan-{up_visi}-ultrasharp-ersgan.jpg')

    


if __name__ == '__main__':
    # image_path = './00084.jpg'
    # upscale_by=4
    # img = upscaler(image_path,upscale_by)
    # img.save('./extra.jpg')
    test()


