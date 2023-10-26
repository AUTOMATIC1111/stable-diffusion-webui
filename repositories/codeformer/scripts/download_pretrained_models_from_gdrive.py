import argparse
import os
from os import path as osp

# from basicsr.utils.download_util import download_file_from_google_drive
import gdown


def download_pretrained_models(method, file_ids):
    save_path_root = f'./weights/{method}'
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        file_url = 'https://drive.google.com/uc?id='+file_id
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(f'{file_name} already exist. Do you want to cover it? Y/N\n')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                gdown.download(file_url, save_path, quiet=False)
                # download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accepts Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            gdown.download(file_url, save_path, quiet=False)
            # download_file_from_google_drive(file_id, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'method',
        type=str,
        help=("Options: 'CodeFormer' 'facelib'. Set to 'all' to download all the models."))
    args = parser.parse_args()

    # file name: file id
    # 'dlib': {
    #     'mmod_human_face_detector-4cb19393.dat': '1qD-OqY8M6j4PWUP_FtqfwUPFPRMu6ubX',
    #     'shape_predictor_5_face_landmarks-c4b1e980.dat': '1vF3WBUApw4662v9Pw6wke3uk1qxnmLdg',
    #     'shape_predictor_68_face_landmarks-fbdc2cb8.dat': '1tJyIVdCHaU6IDMDx86BZCxLGZfsWB8yq'
    # }
    file_ids = {
        'CodeFormer': {
            'codeformer.pth': '1v_E_vZvP-dQPF55Kc5SRCjaKTQXDz-JB'
        },
        'facelib': {
            'yolov5l-face.pth': '131578zMA6B2x8VQHyHfa6GEPtulMCNzV',
            'parsing_parsenet.pth': '16pkohyZZ8ViHGBk3QtVqxLZKzdo466bK'
        }
    }

    if args.method == 'all':
        for method in file_ids.keys():
            download_pretrained_models(method, file_ids[method])
    else:
        download_pretrained_models(args.method, file_ids[args.method])