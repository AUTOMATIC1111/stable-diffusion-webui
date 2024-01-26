import os
import gdown
import requests
from tqdm import tqdm

STABLE_DIFFUSION_DIR = "models/Stable-diffusion/"
CONTROL_NET_MODEL_DIR = "extensions/sd-webui-controlnet/models/"
LORA_MODEL_DIR = "models/Lora/"


def download_model(download_url, des_folder, filename):
    "download_url can be a google drive file ID or a https link"
    filepath = os.path.join(des_folder, filename)
    if not os.path.exists(filepath):
        if "https" not in download_url:
            print(
                f"Download model {filename} from google drive {download_url} to {des_folder}"
            )
            gdown.download(id=download_url, output=des_folder, quiet=False)
        else:
            print(
                f"Download model {filename} from {download_url} to {des_folder}"
            )
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kb
            with open(filepath, "wb") as file, tqdm(
                total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)
            print("\nDownload complete.")


revAnimated_v122EOL_url = "1GGVM1a3jSQdGLszOZRi-aJkP2Kwq5opd"
download_model(
    revAnimated_v122EOL_url,
    STABLE_DIFFUSION_DIR,
    "revAnimated_v122EOL.safetensors",
)

blindbox_v1_mix_url = "1NTBN4b4gq2zl8cM1ulcSkEvAa4sXsoth"
download_model(
    blindbox_v1_mix_url, LORA_MODEL_DIR, "blindbox_v1_mix.safetensors"
)

ip_adapter_faceid_plus_sd15_url = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin"
download_model(
    ip_adapter_faceid_plus_sd15_url,
    CONTROL_NET_MODEL_DIR,
    "ip-adapter-faceid-plus_sd15.bin",
)

dreamshaper_7_url = "1uRIdWeD3mi_FQ6u3-o0EsPtIgWSk2qAF"
download_model(
    dreamshaper_7_url, STABLE_DIFFUSION_DIR, "dreamshaper_7.safetensors"
)
