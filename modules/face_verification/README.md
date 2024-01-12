
![logo](https://yk-website-images.s3.eu-west-1.amazonaws.com/LogoV4_TRANSPARENT.png?)

# YouFace API: Python SDK & Sample

[![PyPi Version](https://img.shields.io/pypi/v/yk_face.svg)](https://pypi.org/project/yk-face/)
[![License](https://img.shields.io/pypi/l/yk_face.svg)](https://github.com/dev-yoonik/YK-Face-Python/blob/master/LICENSE)

This repository contains the Python SDK for the YouFace API, an offering within [Youverse Services](https://www.youverse.id)

## Getting started

Install the module using [pip](https://pypi.python.org/pypi/pip/):

```bash
pip install yk_face
```

Use it:

```python
import yk_face as YKF

KEY = 'subscription key'  # Replace with a valid Subscription Key here.
YKF.Key.set(KEY)

BASE_URL = 'YouFace API URL'  # Replace with a valid URL for YouFace API.
YKF.BaseUrl.set(BASE_URL)

img_file_path = 'image path'  # Replace with a valid image file path here.
detected_faces = YKF.face.process(img_file_path)
print(f'Detected faces: {detected_faces}')
```

### Installing from the source code

```bash
python setup.py install
```

## Running the sample

A sample python script is also provided. Please check the sample directory in this repository.

## YouFace API Details

For a complete specification of our Face API please check the [swagger file](https://dev-yoonik.github.io/YK-Face-Documentation/).


## Contact & Support

For more information and trial licenses please [contact us](mailto:tech@youverse.id) or join us at our [discord community](https://discord.gg/SqHVQUFNtN).




