# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import glob
import os
import json
import re

from tqdm import tqdm

PATTERN_HAIR_LENGTH = re.compile(r', (long|short|medium) hair, ')
PATTERN_HAIR_CUT = re.compile(r', (bob|hime) cut, ')
PATTERN_HAIR = re.compile(r', ([\w\-]+) hair, ')
PATTERN_WORD = re.compile(r', ([\w\-]+|hair ornament), ')

# 複数人がいるとき、複数の髪色や目の色が定義されていれば削除する
PATTERNS_REMOVE_IN_MULTI = [
    PATTERN_HAIR_LENGTH,
    PATTERN_HAIR_CUT,
    re.compile(r', [\w\-]+ eyes, '),
    re.compile(r', ([\w\-]+ sleeves|sleeveless), '),
    # 複数の髪型定義がある場合は削除する
    re.compile(
        r', (ponytail|braid|ahoge|twintails|[\w\-]+ bun|single hair bun|single side bun|two side up|two tails|[\w\-]+ braid|sidelocks), '),
]


def clean_tags(image_key, tags):
  # replace '_' to ' '
  tags = tags.replace('^_^', '^@@@^')
  tags = tags.replace('_', ' ')
  tags = tags.replace('^@@@^', '^_^')

  # remove rating: deepdanbooruのみ
  tokens = tags.split(", rating")
  if len(tokens) == 1:
    # WD14 taggerのときはこちらになるのでメッセージは出さない
    # print("no rating:")
    # print(f"{image_key} {tags}")
    pass
  else:
    if len(tokens) > 2:
      print("multiple ratings:")
      print(f"{image_key} {tags}")
    tags = tokens[0]

  tags = ", " + tags.replace(", ", ", , ") + ", "     # カンマ付きで検索をするための身も蓋もない対策
  
  # 複数の人物がいる場合は髪色等のタグを削除する
  if 'girls' in tags or 'boys' in tags:
    for pat in PATTERNS_REMOVE_IN_MULTI:
      found = pat.findall(tags)
      if len(found) > 1:                        # 二つ以上、タグがある
        tags = pat.sub("", tags)

    # 髪の特殊対応
    srch_hair_len = PATTERN_HAIR_LENGTH.search(tags)   # 髪の長さタグは例外なので避けておく（全員が同じ髪の長さの場合）
    if srch_hair_len:
      org = srch_hair_len.group()
      tags = PATTERN_HAIR_LENGTH.sub(", @@@, ", tags)

    found = PATTERN_HAIR.findall(tags)
    if len(found) > 1:
      tags = PATTERN_HAIR.sub("", tags)

    if srch_hair_len:
      tags = tags.replace(", @@@, ", org)                   # 戻す

  # white shirtとshirtみたいな重複タグの削除
  found = PATTERN_WORD.findall(tags)
  for word in found:
    if re.search(f", ((\w+) )+{word}, ", tags):
      tags = tags.replace(f", {word}, ", "")

  tags = tags.replace(", , ", ", ")
  assert tags.startswith(", ") and tags.endswith(", ")
  tags = tags[2:-2]
  return tags


# 上から順に検索、置換される
# ('置換元文字列', '置換後文字列')
CAPTION_REPLACEMENTS = [
    ('anime anime', 'anime'),
    ('young ', ''),
    ('anime girl', 'girl'),
    ('cartoon female', 'girl'),
    ('cartoon lady', 'girl'),
    ('cartoon character', 'girl'),      # a or ~s
    ('cartoon woman', 'girl'),
    ('cartoon women', 'girls'),
    ('cartoon girl', 'girl'),
    ('anime female', 'girl'),
    ('anime lady', 'girl'),
    ('anime character', 'girl'),      # a or ~s
    ('anime woman', 'girl'),
    ('anime women', 'girls'),
    ('lady', 'girl'),
    ('female', 'girl'),
    ('woman', 'girl'),
    ('women', 'girls'),
    ('people', 'girls'),
    ('person', 'girl'),
    ('a cartoon figure', 'a figure'),
    ('a cartoon image', 'an image'),
    ('a cartoon picture', 'a picture'),
    ('an anime cartoon image', 'an image'),
    ('a cartoon anime drawing', 'a drawing'),
    ('a cartoon drawing', 'a drawing'),
    ('girl girl', 'girl'),
]


def clean_caption(caption):
  for rf, rt in CAPTION_REPLACEMENTS:
    replaced = True
    while replaced:
      bef = caption
      caption = caption.replace(rf, rt)
      replaced = bef != caption
  return caption


def main(args):
  if os.path.exists(args.in_json):
    print(f"loading existing metadata: {args.in_json}")
    with open(args.in_json, "rt", encoding='utf-8') as f:
      metadata = json.load(f)
  else:
    print("no metadata / メタデータファイルがありません")
    return

  print("cleaning captions and tags.")
  image_keys = list(metadata.keys())
  for image_key in tqdm(image_keys):
    tags = metadata[image_key].get('tags')
    if tags is None:
      print(f"image does not have tags / メタデータにタグがありません: {image_key}")
    else:
      org = tags
      tags = clean_tags(image_key, tags)
      metadata[image_key]['tags'] = tags
      if args.debug and org != tags:
        print("FROM: " + org)
        print("TO:   " + tags)

    caption = metadata[image_key].get('caption')
    if caption is None:
      print(f"image does not have caption / メタデータにキャプションがありません: {image_key}")
    else:
      org = caption
      caption = clean_caption(caption)
      metadata[image_key]['caption'] = caption
      if args.debug and org != caption:
        print("FROM: " + org)
        print("TO:   " + caption)

  # metadataを書き出して終わり
  print(f"writing metadata: {args.out_json}")
  with open(args.out_json, "wt", encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
  print("done!")


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  # parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
  parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
  parser.add_argument("--debug", action="store_true", help="debug mode")

  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args, unknown = parser.parse_known_args()
  if len(unknown) == 1:
    print("WARNING: train_data_dir argument is removed. This script will not work with three arguments in future. Please specify two arguments: in_json and out_json.")
    print("All captions and tags in the metadata are processed.")
    print("警告: train_data_dir引数は不要になりました。将来的には三つの引数を指定すると動かなくなる予定です。読み込み元のメタデータと書き出し先の二つの引数だけ指定してください。")
    print("メタデータ内のすべてのキャプションとタグが処理されます。")
    args.in_json = args.out_json
    args.out_json = unknown[0]
  elif len(unknown) > 0:
    raise ValueError(f"error: unrecognized arguments: {unknown}")

  main(args)
