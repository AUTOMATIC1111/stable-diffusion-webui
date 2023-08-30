import opennsfw2 as n2
from opennsfw2._model import make_open_nsfw_model
from opennsfw2._image import preprocess_image, Preprocessing
import numpy as np

model = make_open_nsfw_model()

def probability(image):
    nsimage = preprocess_image(image, Preprocessing.YAHOO)
    nsfw_probability = float(model.predict(np.expand_dims(nsimage, 0), batch_size=1)[0][1])
    return nsfw_probability