import argparse
import glob
import os
import json
import random

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip.blip import blip_decoder
import library.train_util as train_util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMAGE_SIZE = 384

# 正方形でいいのか？　という気がするがソースがそうなので
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)


# 共通化したいが微妙に処理が異なる……
class ImageLoadingTransformDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor = IMAGE_TRANSFORM(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    # fix the seed for reproducibility
    seed = args.seed  # + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists("blip"):
        args.train_data_dir = os.path.abspath(args.train_data_dir)  # convert to absolute path

        cwd = os.getcwd()
        print("Current Working Directory is: ", cwd)
        os.chdir("finetune")

    print(f"load images from {args.train_data_dir}")
    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    print(f"found {len(image_paths)} images.")

    print(f"loading BLIP caption: {args.caption_weights}")
    model = blip_decoder(pretrained=args.caption_weights, image_size=IMAGE_SIZE, vit="large", med_config="./blip/med_config.json")
    model.eval()
    model = model.to(DEVICE)
    print("BLIP loaded")

    # captioningする
    def run_batch(path_imgs):
        imgs = torch.stack([im for _, im in path_imgs]).to(DEVICE)

        with torch.no_grad():
            if args.beam_search:
                captions = model.generate(
                    imgs, sample=False, num_beams=args.num_beams, max_length=args.max_length, min_length=args.min_length
                )
            else:
                captions = model.generate(
                    imgs, sample=True, top_p=args.top_p, max_length=args.max_length, min_length=args.min_length
                )

        for (image_path, _), caption in zip(path_imgs, captions):
            with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding="utf-8") as f:
                f.write(caption + "\n")
                if args.debug:
                    print(image_path, caption)

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingTransformDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            img_tensor, image_path = data
            if img_tensor is None:
                try:
                    raw_image = Image.open(image_path)
                    if raw_image.mode != "RGB":
                        raw_image = raw_image.convert("RGB")
                    img_tensor = IMAGE_TRANSFORM(raw_image)
                except Exception as e:
                    print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue

            b_imgs.append((image_path, img_tensor))
            if len(b_imgs) >= args.batch_size:
                run_batch(b_imgs)
                b_imgs.clear()
    if len(b_imgs) > 0:
        run_batch(b_imgs)

    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument(
        "--caption_weights",
        type=str,
        default="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
        help="BLIP caption weights (model_large_caption.pth) / BLIP captionの重みファイル(model_large_caption.pth)",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）",
    )
    parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument(
        "--beam_search",
        action="store_true",
        help="use beam search (default Nucleus sampling) / beam searchを使う（このオプション未指定時はNucleus sampling）",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument("--num_beams", type=int, default=1, help="num of beams in beam search /beam search時のビーム数（多いと精度が上がるが時間がかかる）")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
    parser.add_argument("--max_length", type=int, default=75, help="max length of caption / captionの最大長")
    parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
    parser.add_argument("--seed", default=42, type=int, help="seed for reproducibility / 再現性を確保するための乱数seed")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    # スペルミスしていたオプションを復元する
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    main(args)
