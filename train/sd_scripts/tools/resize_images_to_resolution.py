import glob
import os
import cv2
import argparse
import shutil
import math
from PIL import Image
import numpy as np


def resize_images(src_img_folder, dst_img_folder, max_resolution="512x512", divisible_by=2, interpolation=None, save_as_png=False, copy_associated_files=False):
  # Split the max_resolution string by "," and strip any whitespaces
  max_resolutions = [res.strip() for res in max_resolution.split(',')]

  # # Calculate max_pixels from max_resolution string
  # max_pixels = int(max_resolution.split("x")[0]) * int(max_resolution.split("x")[1])

  # Create destination folder if it does not exist
  if not os.path.exists(dst_img_folder):
    os.makedirs(dst_img_folder)

  # Select interpolation method
  if interpolation == 'lanczos4':
    cv2_interpolation = cv2.INTER_LANCZOS4
  elif interpolation == 'cubic':
    cv2_interpolation = cv2.INTER_CUBIC
  else:
    cv2_interpolation = cv2.INTER_AREA

  # Iterate through all files in src_img_folder
  img_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")                   # copy from train_util.py
  for filename in os.listdir(src_img_folder):
    # Check if the image is png, jpg or webp etc...
    if not filename.endswith(img_exts):
      # Copy the file to the destination folder if not png, jpg or webp etc (.txt or .caption or etc.)
      shutil.copy(os.path.join(src_img_folder, filename), os.path.join(dst_img_folder, filename))
      continue

    # Load image
    # img = cv2.imread(os.path.join(src_img_folder, filename))
    image = Image.open(os.path.join(src_img_folder, filename))
    if not image.mode == "RGB":
      image = image.convert("RGB")
    img = np.array(image, np.uint8)

    base, _ = os.path.splitext(filename)
    for max_resolution in max_resolutions:
      # Calculate max_pixels from max_resolution string
      max_pixels = int(max_resolution.split("x")[0]) * int(max_resolution.split("x")[1])

      # Calculate current number of pixels
      current_pixels = img.shape[0] * img.shape[1]

      # Check if the image needs resizing
      if current_pixels > max_pixels:
        # Calculate scaling factor
        scale_factor = max_pixels / current_pixels

        # Calculate new dimensions
        new_height = int(img.shape[0] * math.sqrt(scale_factor))
        new_width = int(img.shape[1] * math.sqrt(scale_factor))

        # Resize image
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2_interpolation)
      else:
        new_height, new_width = img.shape[0:2]

      # Calculate the new height and width that are divisible by divisible_by (with/without resizing)
      new_height = new_height if new_height % divisible_by == 0 else new_height - new_height % divisible_by
      new_width = new_width if new_width % divisible_by == 0 else new_width - new_width % divisible_by

      # Center crop the image to the calculated dimensions
      y = int((img.shape[0] - new_height) / 2)
      x = int((img.shape[1] - new_width) / 2)
      img = img[y:y + new_height, x:x + new_width]

      # Split filename into base and extension
      new_filename = base + '+' + max_resolution + ('.png' if save_as_png else '.jpg')

      # Save resized image in dst_img_folder
      # cv2.imwrite(os.path.join(dst_img_folder, new_filename), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
      image = Image.fromarray(img)
      image.save(os.path.join(dst_img_folder, new_filename), quality=100)

      proc = "Resized" if current_pixels > max_pixels else "Saved"
      print(f"{proc} image: {filename} with size {img.shape[0]}x{img.shape[1]} as {new_filename}")

    # If other files with same basename, copy them with resolution suffix
    if copy_associated_files:
      asoc_files = glob.glob(os.path.join(src_img_folder, base + ".*"))
      for asoc_file in asoc_files:
        ext = os.path.splitext(asoc_file)[1]
        if ext in img_exts:
          continue
        for max_resolution in max_resolutions:
          new_asoc_file = base + '+' + max_resolution + ext
          print(f"Copy {asoc_file} as {new_asoc_file}")
          shutil.copy(os.path.join(src_img_folder, asoc_file), os.path.join(dst_img_folder, new_asoc_file))


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description='Resize images in a folder to a specified max resolution(s) / 指定されたフォルダ内の画像を指定した最大画像サイズ（面積）以下にアスペクト比を維持したままリサイズします')
  parser.add_argument('src_img_folder', type=str, help='Source folder containing the images / 元画像のフォルダ')
  parser.add_argument('dst_img_folder', type=str, help='Destination folder to save the resized images / リサイズ後の画像を保存するフォルダ')
  parser.add_argument('--max_resolution', type=str,
                      help='Maximum resolution(s) in the format "512x512,384x384, etc, etc" / 最大画像サイズをカンマ区切りで指定 ("512x512,384x384, etc, etc" など)', default="512x512,384x384,256x256,128x128")
  parser.add_argument('--divisible_by', type=int,
                      help='Ensure new dimensions are divisible by this value / リサイズ後の画像のサイズをこの値で割り切れるようにします', default=1)
  parser.add_argument('--interpolation', type=str, choices=['area', 'cubic', 'lanczos4'],
                      default='area', help='Interpolation method for resizing / リサイズ時の補完方法')
  parser.add_argument('--save_as_png', action='store_true', help='Save as png format / png形式で保存')
  parser.add_argument('--copy_associated_files', action='store_true',
                      help='Copy files with same base name to images (captions etc) / 画像と同じファイル名（拡張子を除く）のファイルもコピーする')

  return parser


def main():
  parser = setup_parser()

  args = parser.parse_args()
  resize_images(args.src_img_folder, args.dst_img_folder, args.max_resolution,
                args.divisible_by, args.interpolation, args.save_as_png, args.copy_associated_files)


if __name__ == '__main__':
  main()
