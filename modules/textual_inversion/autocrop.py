import cv2
from collections import defaultdict
from math import log, sqrt
import numpy as np
from PIL import Image, ImageDraw

GREEN = "#0F0"
BLUE = "#00F"
RED = "#F00"

def crop_image(im, settings):
  """ Intelligently crop an image to the subject matter """
  if im.height > im.width:
      im = im.resize((settings.crop_width, settings.crop_height * im.height // im.width))
  else:
      im = im.resize((settings.crop_width * im.width // im.height, settings.crop_height))

  focus = focal_point(im, settings)

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

  if settings.annotate_image:
    d = ImageDraw.Draw(im)
    rect = list(crop)
    rect[2] -= 1
    rect[3] -= 1
    d.rectangle(rect, outline=GREEN)
    if settings.destop_view_image:
      im.show()

  return im.crop(tuple(crop))

def focal_point(im, settings):
    corner_points = image_corner_points(im, settings)
    entropy_points = image_entropy_points(im, settings)
    face_points = image_face_points(im, settings)

    total_points = len(corner_points) + len(entropy_points) + len(face_points)

    corner_weight = settings.corner_points_weight
    entropy_weight = settings.entropy_points_weight
    face_weight = settings.face_points_weight

    weight_pref_total = corner_weight + entropy_weight + face_weight

    # weight things
    pois = []
    if weight_pref_total == 0 or total_points == 0: 
      return pois

    pois.extend(
      [ PointOfInterest( p.x, p.y, weight=p.weight * ( (corner_weight/weight_pref_total) / (len(corner_points)/total_points) )) for p in corner_points ]
    )
    pois.extend(
      [ PointOfInterest( p.x, p.y, weight=p.weight * ( (entropy_weight/weight_pref_total) / (len(entropy_points)/total_points) )) for p in entropy_points ]
    )
    pois.extend(
      [ PointOfInterest( p.x, p.y, weight=p.weight * ( (face_weight/weight_pref_total) / (len(face_points)/total_points) )) for p in face_points ]
    )

    if settings.annotate_image:
      d = ImageDraw.Draw(im)

    average_point = poi_average(pois, settings, im=im)

    if settings.annotate_image:
      d.ellipse([average_point.x - 25, average_point.y - 25, average_point.x + 25, average_point.y + 25], outline=GREEN)
      
    return average_point


def image_face_points(im, settings):
    np_im = np.array(im)
    gray = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

    minsize = int(min(im.width, im.height) * 0.15) # at least N percent of the smallest side
    faces = classifier.detectMultiScale(gray, scaleFactor=1.05,
      minNeighbors=5, minSize=(minsize, minsize), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) == 0:
      return []

    rects = [[f[0], f[1], f[0] + f[2], f[1] + f[3]] for f in faces]
    if settings.annotate_image:
      for f in rects:
        d = ImageDraw.Draw(im)
        d.rectangle(f, outline=RED)
    
    return [PointOfInterest((r[0] +r[2]) // 2, (r[1] + r[3]) // 2) for r in rects]


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
        minDistance=min(grayscale.width, grayscale.height)*0.07,
        useHarrisDetector=False,
    )

    if points is None:
        return []

    focal_points = []
    for point in points:
        x, y = point.ravel()
        focal_points.append(PointOfInterest(x, y))

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

    return [PointOfInterest(x_mid, y_mid)]


def image_entropy(im):
    # greyscale image entropy
    band = np.asarray(im.convert("1"))
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


def poi_average(pois, settings, im=None):
    weight = 0.0
    x = 0.0
    y = 0.0
    for pois in pois:
        if settings.annotate_image and im is not None:
          w = 4 * 0.5 * sqrt(pois.weight)
          d = ImageDraw.Draw(im)
          d.ellipse([
            pois.x - w, pois.y - w,
            pois.x + w, pois.y + w ], fill=BLUE)
        weight += pois.weight
        x += pois.x * pois.weight
        y += pois.y * pois.weight
    avg_x = round(x / weight)
    avg_y = round(y / weight)

    return PointOfInterest(avg_x, avg_y)


class PointOfInterest:
  def __init__(self, x, y, weight=1.0):
    self.x = x
    self.y = y
    self.weight = weight


class Settings:
  def __init__(self, crop_width=512, crop_height=512, corner_points_weight=0.5, entropy_points_weight=0.5, face_points_weight=0.5, annotate_image=False):
    self.crop_width = crop_width
    self.crop_height = crop_height
    self.corner_points_weight = corner_points_weight
    self.entropy_points_weight = entropy_points_weight
    self.face_points_weight = entropy_points_weight
    self.annotate_image = annotate_image
    self.destop_view_image = False