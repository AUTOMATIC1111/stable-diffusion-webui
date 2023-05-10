import argparse
import os
import json

from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

import library.model_util as model_util
import library.train_util as train_util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_latents(vae, images, weight_dtype):
    img_tensors = [IMAGE_TRANSFORMS(image) for image in images]
    img_tensors = torch.stack(img_tensors)
    img_tensors = img_tensors.to(DEVICE, weight_dtype)
    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().float().to("cpu").numpy()
    return latents


def get_npz_filename_wo_ext(data_dir, image_key, is_full_path, flip, recursive):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if flip:
        base_name += "_flip"

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name)
    else:
        return os.path.join(data_dir, base_name)


def main(args):
    # assert args.bucket_reso_steps % 8 == 0, f"bucket_reso_steps must be divisible by 8 / bucket_reso_stepは8で割り切れる必要があります"
    if args.bucket_reso_steps % 8 > 0:
        print(f"resolution of buckets in training time is a multiple of 8 / 学習時の各bucketの解像度は8単位になります")

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, args.recursive)]
    print(f"found {len(image_paths)} images.")

    if os.path.exists(args.in_json):
        print(f"loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        print(f"no metadata / メタデータファイルがありません: {args.in_json}")
        return

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = model_util.load_vae(args.model_name_or_path, weight_dtype)
    vae.eval()
    vae.to(DEVICE, dtype=weight_dtype)

    # bucketのサイズを計算する
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert len(max_reso) == 2, f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

    bucket_manager = train_util.BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )
    if not args.bucket_no_upscale:
        bucket_manager.make_buckets()
    else:
        print(
            "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
        )

    # 画像をひとつずつ適切なbucketに割り当てながらlatentを計算する
    img_ar_errors = []

    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                latents = get_latents(vae, [img for _, img in bucket], weight_dtype)
                assert (
                    latents.shape[2] == bucket[0][1].shape[0] // 8 and latents.shape[3] == bucket[0][1].shape[1] // 8
                ), f"latent shape {latents.shape}, {bucket[0][1].shape}"

                for (image_key, _), latent in zip(bucket, latents):
                    npz_file_name = get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, False, args.recursive)
                    np.savez(npz_file_name, latent)

                # flip
                if args.flip_aug:
                    latents = get_latents(vae, [img[:, ::-1].copy() for _, img in bucket], weight_dtype)  # copyがないとTensor変換できない

                    for (image_key, _), latent in zip(bucket, latents):
                        npz_file_name = get_npz_filename_wo_ext(
                            args.train_data_dir, image_key, args.full_path, True, args.recursive
                        )
                        np.savez(npz_file_name, latent)
                else:
                    # remove existing flipped npz
                    for image_key, _ in bucket:
                        npz_file_name = (
                            get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, True, args.recursive) + ".npz"
                        )
                        if os.path.isfile(npz_file_name):
                            print(f"remove existing flipped npz / 既存のflipされたnpzファイルを削除します: {npz_file_name}")
                            os.remove(npz_file_name)

                bucket.clear()

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        dataset = train_util.ImageLoadingDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    bucket_counts = {}
    for data_entry in tqdm(data, smoothing=0.0):
        if data_entry[0] is None:
            continue

        img_tensor, image_path = data_entry[0]
        if img_tensor is not None:
            image = transforms.functional.to_pil_image(img_tensor)
        else:
            try:
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                continue

        image_key = image_path if args.full_path else os.path.splitext(os.path.basename(image_path))[0]
        if image_key not in metadata:
            metadata[image_key] = {}

        # 本当はこのあとの部分もDataSetに持っていけば高速化できるがいろいろ大変

        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        # メタデータに記録する解像度はlatent単位とするので、8単位で切り捨て
        metadata[image_key]["train_resolution"] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)

        if not args.bucket_no_upscale:
            # upscaleを行わないときには、resize後のサイズは、bucketのサイズと、縦横どちらかが同じであることを確認する
            assert (
                resized_size[0] == reso[0] or resized_size[1] == reso[1]
            ), f"internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
            assert (
                resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
            ), f"internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

        assert (
            resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
        ), f"internal error resized size is small: {resized_size}, {reso}"

        # 既に存在するファイルがあればshapeを確認して同じならskipする
        if args.skip_existing:
            npz_files = [get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, False, args.recursive) + ".npz"]
            if args.flip_aug:
                npz_files.append(
                    get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, True, args.recursive) + ".npz"
                )

            found = True
            for npz_file in npz_files:
                if not os.path.exists(npz_file):
                    found = False
                    break

                dat = np.load(npz_file)["arr_0"]
                if dat.shape[1] != reso[1] // 8 or dat.shape[2] != reso[0] // 8:  # latentsのshapeを確認
                    found = False
                    break
            if found:
                continue

        # 画像をリサイズしてトリミングする
        # PILにinter_areaがないのでcv2で……
        image = np.array(image)
        if resized_size[0] != image.shape[1] or resized_size[1] != image.shape[0]:  # リサイズ処理が必要？
            image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)

        if resized_size[0] > reso[0]:
            trim_size = resized_size[0] - reso[0]
            image = image[:, trim_size // 2 : trim_size // 2 + reso[0]]

        if resized_size[1] > reso[1]:
            trim_size = resized_size[1] - reso[1]
            image = image[trim_size // 2 : trim_size // 2 + reso[1]]

        assert (
            image.shape[0] == reso[1] and image.shape[1] == reso[0]
        ), f"internal error, illegal trimmed size: {image.shape}, {reso}"

        # # debug
        # cv2.imwrite(f"r:\\test\\img_{len(img_ar_errors)}.jpg", image[:, :, ::-1])

        # バッチへ追加
        bucket_manager.add_image(reso, (image_key, image))

        # バッチを推論するか判定して推論する
        process_batch(False)

    # 残りを処理する
    process_batch(True)

    bucket_manager.sort()
    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            print(f"bucket {i} {reso}: {count}")
    img_ar_errors = np.array(img_ar_errors)
    print(f"mean ar error: {np.mean(img_ar_errors)}")

    # metadataを書き出して終わり
    print(f"writing metadata: {args.out_json}")
    with open(args.out_json, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("model_name_or_path", type=str, help="model name or path to encode latents / latentを取得するためのモデル")
    parser.add_argument("--v2", action="store_true", help="not used (for backward compatibility) / 使用されません（互換性のため残してあります）")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="512,512",
        help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）",
    )
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最小解像度")
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale", action="store_true", help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度"
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）",
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="flip augmentation, save latents for flipped images / 左右反転した画像もlatentを取得、保存する"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
