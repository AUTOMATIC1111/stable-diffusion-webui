import os
from modules import paths
from basicsr.utils.download_util import load_file_from_url


urls = {
    'CodeFormer': {
        'codeformer.pth': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    },
    'facelib': {
        # 'yolov5l-face.pth': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/yolov5l-face.pth',
        'detection_Resnet50_Final.pth': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing_parsenet.pth': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
    }
}


def download_pretrained_models(file_urls):
    model_dir = os.path.join(paths.models_path, 'Codeformer')
    for file_name, file_url in file_urls.items():
        load_file_from_url(url=file_url, model_dir=model_dir, progress=True, file_name=file_name)


if __name__ == '__main__':
    for method in urls.keys():
        download_pretrained_models(urls[method])
