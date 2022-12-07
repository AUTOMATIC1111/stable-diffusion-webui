import cv2
import requests
import os
from collections import defaultdict
from math import log, sqrt
import numpy as np
from PIL import Image, ImageDraw

GREEN = "#0F0"
BLUE = "#00F"
RED = "#F00"


def crop_image(im, settings):
  """ Intelligently crop an image to the subject matter """

  scale_by = 1
  if is_landscape(im.width, im.height):
    scale_by = settings.crop_height / im.height
  elif is_portrait(im.width, im.height):
    scale_by = settings.crop_width / im.width
  elif is_square(im.width, im.height):
    if is_square(settings.crop_width, settings.crop_height):
      scale_by = settings.crop_width / im.width
    elif is_landscape(settings.crop_width, settings.crop_height):
      scale_by = settings.crop_width / im.width
    elif is_portrait(settings.crop_width, settings.crop_height):
      scale_by = settings.crop_height / im.height

  im = im.resize((int(im.width * scale_by), int(im.height * scale_by)))
  im_debug = im.copy()

  focus = focal_point(im_debug, settings)

  # take the focal point and turn it into crop coordinates that try to center over the focal
  # point but then get adjusted back into the frame
  y_half = int(settings.crop_height / 2)
  x_half = int(settings.crop_width / 2)

  x1 = focus.x - x_half
  if x1 < 0:
      x1 = 0
  elif x1 + settings.crop_width > im.width:
      x1 = im.width - settings.crop_width

  y1 = focus.y - y_half
  if y1 < 0:
      y1 = 0
  elif y1 + settings.crop_height > im.height:
      y1 = im.height - settings.crop_height

  x2 = x1 + settings.crop_width
  y2 = y1 + settings.crop_height

  crop = [x1, y1, x2, y2]

  results = []

  results.append(im.crop(tuple(crop)))

  if settings.annotate_image:
    d = ImageDraw.Draw(im_debug)
    rect = list(crop)
    rect[2] -= 1
    rect[3] -= 1
    d.rectangle(rect, outline=GREEN)
    results.append(im_debug)
    if settings.destop_view_image:
      im_debug.show()

  return results

def focal_point(im, settings):
    corner_points = image_corner_points(im, settings) if settings.corner_points_weight > 0 else []
    entropy_points = image_entropy_points(im, settings) if settings.entropy_points_weight > 0 else []
    face_points = image_face_points(im, settings) if settings.face_points_weight > 0 else []

    pois = []

    weight_pref_total = 0
    if len(corner_points) > 0:
      weight_pref_total += settings.corner_points_weight
    if len(entropy_points) > 0:
      weight_pref_total += settings.entropy_points_weight
    if len(face_points) > 0:
      weight_pref_total += settings.face_points_weight

    corner_centroid = None
    if len(corner_points) > 0:
      corner_centroid = centroid(corner_points)
      corner_centroid.weight = settings.corner_points_weight / weight_pref_total 
      pois.append(corner_centroid)

    entropy_centroid = None
    if len(entropy_points) > 0:
      entropy_centroid = centroid(entropy_points)
      entropy_centroid.weight = settings.entropy_points_weight / weight_pref_total
      pois.append(entropy_centroid)

    face_centroid = None
    if len(face_points) > 0:
      face_centroid = centroid(face_points)
      face_centroid.weight = settings.face_points_weight / weight_pref_total 
      pois.append(face_centroid)

    average_point = poi_average(pois, settings)

    if settings.annotate_image:
      d = ImageDraw.Draw(im)
      max_size = min(im.width, im.height) * 0.07
      if corner_centroid is not None:
        color = BLUE
        box = corner_centroid.bounding(max_size * corner_centroid.weight)
        d.text((box[0], box[1]-15), "Edge: %.02f" % corner_centroid.weight, fill=color)
        d.ellipse(box, outline=color)
        if len(corner_points) > 1:
          for f in corner_points:
            d.rectangle(f.bounding(4), outline=color)
      if entropy_centroid is not None:
        color = "#ff0"
        box = entropy_centroid.bounding(max_size * entropy_centroid.weight)
        d.text((box[0], box[1]-15), "Entropy: %.02f" % entropy_centroid.weight, fill=color)
        d.ellipse(box, outline=color)
        if len(entropy_points) > 1:
          for f in entropy_points:
            d.rectangle(f.bounding(4), outline=color)
      if face_centroid is not None:
        color = RED
        box = face_centroid.bounding(max_size * face_centroid.weight)
        d.text((box[0], box[1]-15), "Face: %.02f" % face_centroid.weight, fill=color)
        d.ellipse(box, outline=color)
        if len(face_points) > 1:
          for f in face_points:
            d.rectangle(f.bounding(4), outline=color)

      d.ellipse(average_point.bounding(max_size), outline=GREEN)
      
    return average_point


def image_face_points(im, settings):
    if settings.dnn_model_path is not None:
      detector = cv2.FaceDetectorYN.create(
          settings.dnn_model_path,
          "",
          (im.width, im.height),
          0.9, # score threshold
          0.3, # nms threshold
          5000 # keep top k before nms
      )
      faces = detector.detect(np.array(im))
      results = []
      if faces[1] is not None:
        for face in faces[1]:
          x = face[0]
          y = face[1]
          w = face[2]
          h = face[3]
          results.append(
            PointOfInterest(
              int(x + (w * 0.5)), # face focus left/right is center
              int(y + (h * 0.33)), # face focus up/down is close to the top of the head
              size = w,
              weight = 1/len(faces[1])
            )
          )
      return results
    else:
      np_im = np.array(im)
      gray = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)

      tries = [
        [ f'{cv2.data.haarcascades}haarcascade_eye.xml', 0.01 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_profileface.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_alt.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_frontalface_alt_tree.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml', 0.05 ],
        [ f'{cv2.data.haarcascades}haarcascade_upperbody.xml', 0.05 ]
      ]
      for t in tries:
        classifier = cv2.CascadeClassifier(t[0])
        minsize = int(min(im.width, im.height) * t[1]) # at least N percent of the smallest side
        try:
          faces = classifier.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=7, minSize=(minsize, minsize), flags=cv2.CASCADE_SCALE_IMAGE)
        except:
          continue

        if len(faces) > 0:
          rects = [[f[0], f[1], f[0] + f[2], f[1] + f[3]] for f in faces]
          return [PointOfInterest((r[0] +r[2]) // 2, (r[1] + r[3]) // 2, size=abs(r[0]-r[2]), weight=1/len(rects)) for r in rects]
    return []


def image_corner_points(im, settings):
    grayscale = im.convert("L")

    # naive attempt at preventing focal points from collecting at watermarks near the bottom
    gd = ImageDraw.Draw(grayscale)
    gd.rectangle([0, im.height*.9, im.width, im.height], fill="#999")

    np_im = np.array(grayscale)

    points = cv2.goodFeaturesToTrack(
        np_im,
        maxCorners=100,
        qualityLevel=0.04,
        minDistance=min(grayscale.width, grayscale.height)*0.06,
        useHarrisDetector=False,
    )

    if points is None:
        return []

    focal_points = []
    for point in points:
      x, y = point.ravel()
      focal_points.append(PointOfInterest(x, y, size=4, weight=1/len(points)))

    return focal_points


def image_entropy_points(im, settings):
    landscape = im.height < im.width
    portrait = im.height > im.width
    if landscape:
      move_idx = [0, 2]
      move_max = im.size[0]
    elif portrait:
      move_idx = [1, 3]
      move_max = im.size[1]
    else:
      return []

    e_max = 0
    crop_current = [0, 0, settings.crop_width, settings.crop_height]
    crop_best = crop_current
    while crop_current[move_idx[1]] < move_max:
        crop = im.crop(tuple(crop_current))
        e = image_entropy(crop)

        if (e > e_max):
          e_max = e
          crop_best = list(crop_current)

        crop_current[move_idx[0]] += 4
        crop_current[move_idx[1]] += 4

    x_mid = int(crop_best[0] + settings.crop_width/2)
    y_mid = int(crop_best[1] + settings.crop_height/2)

    return [PointOfInterest(x_mid, y_mid, size=25, weight=1.0)]


def image_entropy(im):
    # greyscale image entropy
    # band = np.asarray(im.convert("L"))
    band = np.asarray(im.convert("1"), dtype=np.uint8)
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()

def centroid(pois):
  x = [poi.x for poi in pois]
  y = [poi.y for poi in pois]
  return PointOfInterest(sum(x)/len(pois), sum(y)/len(pois))


def poi_average(pois, settings):
    weight = 0.0
    x = 0.0
    y = 0.0
    for poi in pois:
        weight += poi.weight
        x += poi.x * poi.weight
        y += poi.y * poi.weight
    avg_x = round(weight and x / weight)
    avg_y = round(weight and y / weight)

    return PointOfInterest(avg_x, avg_y)


def is_landscape(w, h):
  return w > h


def is_portrait(w, h):
  return h > w


def is_square(w, h):
  return w == h


def download_and_cache_models(dirname):
  download_url = 'https://github.com/opencv/opencv_zoo/blob/91fb0290f50896f38a0ab1e558b74b16bc009428/models/face_detection_yunet/face_detection_yunet_2022mar.onnx?raw=true'
  model_file_name = 'face_detection_yunet.onnx'

  if not os.path.exists(dirname):
    os.makedirs(dirname)

  cache_file = os.path.join(dirname, model_file_name)
  if not os.path.exists(cache_file):
    print(f"downloading face detection model from '{download_url}' to '{cache_file}'")
    response = requests.get(download_url)
    with open(cache_file, "wb") as f:
      f.write(response.content)

  if os.path.exists(cache_file):
    return cache_file
  return None


class PointOfInterest:
  def __init__(self, x, y, weight=1.0, size=10):
    self.x = x
    self.y = y
    self.weight = weight
    self.size = size

  def bounding(self, size):
    return [
      self.x - size//2,
      self.y - size//2,
      self.x + size//2,
      self.y + size//2
    ]


class Settings:
  def __init__(self, crop_width=512, crop_height=512, corner_points_weight=0.5, entropy_points_weight=0.5, face_points_weight=0.5, annotate_image=False, dnn_model_path=None):
    self.crop_width = crop_width
    self.crop_height = crop_height
    self.corner_points_weight = corner_points_weight
    self.entropy_points_weight = entropy_points_weight
    self.face_points_weight = face_points_weight
    self.annotate_image = annotate_image
    self.destop_view_image = False
    self.dnn_model_path = dnn_model_path
