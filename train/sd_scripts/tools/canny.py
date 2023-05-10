import argparse
import cv2


def canny(args):
  img = cv2.imread(args.input)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  canny_img = cv2.Canny(img, args.thres1, args.thres2)
  # canny_img = 255 - canny_img

  cv2.imwrite(args.output, canny_img)
  print("done!")


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, default=None, help="input path")
  parser.add_argument("--output", type=str, default=None, help="output path")
  parser.add_argument("--thres1", type=int, default=32, help="thres1")
  parser.add_argument("--thres2", type=int, default=224, help="thres2")

  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()
  canny(args)
