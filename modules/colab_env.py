from modules import launch_utils
import concurrent.futures
from modules.paths_internal import models_path

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

run = launch_utils.run

def pip_1():
  run('apt-get -y update -qq')
  run('apt-get -y install -qq aria2')   
  run('pip install -qq pycloudflared ngrok tntn fastapi==0.94')
  run('pip install -qq translators chardet openai boto3 aliyun-python-sdk-core aliyun-python-sdk-alimt python-dotenv pyfunctional')

def pip_2():
  run("pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -U")
  run("pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U")
  run(f"!curl -Lo {models_path}/Stable-diffusion/https://huggingface.co/spaces/weo1101/111/resolve/main/chilloutmix_NiPrunedFp32Fix-inpainting.inpainting.safetensors https://huggingface.co/spaces/weo1101/111/resolve/main/chilloutmix_NiPrunedFp32Fix-inpainting.inpainting.safetensors")

def pip_3():
    executor=concurrent.futures.ThreadPoolExecutor(max_workers=3)
    task1=executor.submit(pip_1)
    task2=executor.submit(pip_2)
    concurrent.futures.wait([task1,task2])