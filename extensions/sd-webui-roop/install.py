import os
import subprocess  
import sys
from tqdm import tqdm
import urllib.request

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

models_dir = os.path.abspath("models/roop")
model_url = "https://github.com/dream80/roop_colab/releases/download/v0.0.1/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

try:
    subprocess.run(["pip", "install", "-r", req_file], check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to install requirements: {e.stderr.decode('utf-8')}")
    sys.exit(1)