import os
import gdown
import requests

STABLE_DIFFUSION_DIR = "models/Stable-diffusion/"
CONTROL_NET_MODEL_DIR = "extensions/sd-webui-controlnet/models/"
LORA_MODEL_DIR = "models/Lora/"

revAnimated_v122EOL_url = "1GGVM1a3jSQdGLszOZRi-aJkP2Kwq5opd"
if not os.path.exists(os.path.join(STABLE_DIFFUSION_DIR, "revAnimated_v122EOL.safetensors")):
    print("Downloading revAnimated_v122EOL.safetensors from 1GGVM1a3jSQdGLszOZRi-aJkP2Kwq5opd")
    gdown.download(id=revAnimated_v122EOL_url, output=STABLE_DIFFUSION_DIR, quiet=False)

blindbox_v1_mix_url = "1NTBN4b4gq2zl8cM1ulcSkEvAa4sXsoth"
if not os.path.exists(os.path.join(LORA_MODEL_DIR, "blindbox_v1_mix.safetensors")):
    print(f"Downloading blindbox_v1_mix.safetensors from {blindbox_v1_mix_url}")
    gdown.download(id=blindbox_v1_mix_url, output=LORA_MODEL_DIR, quiet=False)


ip_adapter_faceid_plus_sd15_url = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin"

ip_adapter_faceid_plus_sd15_file = os.path.join(CONTROL_NET_MODEL_DIR, os.path.basename(ip_adapter_faceid_plus_sd15_url))
if not os.path.exists(ip_adapter_faceid_plus_sd15_file):
    print("downloading ip_adapter_faceid_plus_sd15")
    response = requests.get(ip_adapter_faceid_plus_sd15_url)
    with open(ip_adapter_faceid_plus_sd15_file, 'wb') as file:
        file.write(response.content)
        print("downloaded ip_adapter_faceid_plus_sd15")
