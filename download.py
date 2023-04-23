import os
import shutil
import glob
import hashlib
import json
import subprocess


# free space --------------------------------------------------

#remove cache if exists
if os.path.exists("/root/.cache"):
    shutil.rmtree("/root/.cache")

# remove default ckpts
for file in glob.glob("/workspace/v2*"):
    print("Removing " + file)
    try:
        os.remove(file)
    except OSError as e:
        print("Error deleting " + file + ": " + str(e))


# controlnet & models --------------------------------------------------

CONTROLNET_DIR = "/workspace/stable-diffusion-webui/extensions/sd-webui-controlnet"
CONTROLNET_REPO = "https://github.com/Mikubill/sd-webui-controlnet"

if not os.path.exists(CONTROLNET_DIR):

    command = f"git clone {CONTROLNET_REPO} {CONTROLNET_DIR}"
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print("Git Clone:", output.stderr)

models = [
    'control_canny-fp16.safetensors',
    'control_openpose-fp16.safetensors',
    'control_depth-fp16.safetensors',
    'control_hed-fp16.safetensors',
    'control_mlsd-fp16.safetensors',
    'control_normal-fp16.safetensors',
    'control_scribble-fp16.safetensors',
    'control_seg-fp16.safetensors'
]

for model in models:
    if not os.path.exists(os.path.join(CONTROLNET_DIR,'models',model)):
        command = f"curl -Lo {os.path.join(CONTROLNET_DIR,'models',model)} https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/{model}"
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        print("Download Model:", output.stderr)

#embeddings / textual inversions --------------------------------------------------
EMBEDDINGS_DIR = "/workspace/stable-diffusion-webui/embeddings"
embeddings_url = 'https://storage.googleapis.com/ag-diffusion/embeddings/embeddings.zip'
if not os.path.exists(os.path.join(EMBEDDINGS_DIR,'embeddings.zip')):
    command = f"curl -Lo {os.path.join(EMBEDDINGS_DIR,'embeddings.zip')} {embeddings_url}"
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print("Download Embeddings:", output.stderr)
    command = f"unzip embeddings/embeddings.zip -d embeddings/"
    output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print("unzip Embeddings:", output.stderr)



# diffusion checkpoints --------------------------------------------------

models_urls  = [
    # popular ckpts from civit.ai
['https://storage.googleapis.com/ag-diffusion/ckpts/deliberate_v2.safetensors','9aba26abdfcd46073e0a1d42027a3a3bcc969f562d58a03637bf0a0ded6586c9'],
['https://storage.googleapis.com/ag-diffusion/ckpts/dreamshaper_331BakedVae.safetensors','1dceefec0715b23865d23c8145f2c1558bb4402570e6f2857b664d8cada46ae1'],
['https://storage.googleapis.com/ag-diffusion/ckpts/realisticVisionV13_v13.safetensors','c35782bad87a6c8d461fa88861d9ee6974d7a16cc19014bfaf89f586dd42ca63'],
['https://storage.googleapis.com/ag-diffusion/ckpts/CounterfeitV25_25.safetensors','a074b8864e31b8681e40db3dfde0005df7b5309fd2a2f592a2caee59e4591cae'],


# popular ckpts on civit.ai - mature
['https://storage.googleapis.com/ag-diffusion/ckpts/grapefruitHentaiModel_grapefruitv41.safetensors','c590550ea5f3ea3ad9126a264ed27970ce4a14eef134900599a73f00b64c4855'],
['https://storage.googleapis.com/ag-diffusion/ckpts/perfectWorld_v2Baked.safetensors','79e42fb7445bb08cb16e92cfd57f3ab09b57f18b1b8bcb27cb5d5d4e19ac1eec'],
['https://storage.googleapis.com/ag-diffusion/ckpts/chilloutmix_NiPrunedFp32Fix.safetensors','fc2511737a54c5e80b89ab03e0ab4b98d051ab187f92860f3cd664dc9d08b271'],
['https://storage.googleapis.com/ag-diffusion/ckpts/uberRealisticPornMerge_urpmv13.safetensors','f93e6a50acf7c1b1e6d5ccaf92bae207cb5d2222da3ddc5108df4c928d84255a'],

]

# todo check file checkum is correct if not delete and redownload
MODEL_DIR = '/workspace/stable-diffusion-webui/models/Stable-diffusion'
checksumfile = os.path.join(MODEL_DIR,"checksum.json")
saved_checksums = {}
if os.path.exists(checksumfile):
    print("loading saved checksums")
    with open(checksumfile) as fp:
        saved_checksums = json.load(fp)

def verifychecksum(basename, correct_checksum, stored_checksums):
    if not os.path.exists(basename):
        print(f"File {basename} does not exist")
        return False
    if basename in stored_checksums:
        if stored_checksums[basename] == correct_checksum:
            print(f"Stored checksum of {basename} was correct")
            return True
        else:
            print(f"Stored checksum of {basename} was INCORRECT: {stored_checksums[basename]}")
            return False
    else:
        hash_algorithm = hashlib.sha256()
        with open(basename, "rb") as f:
            while chunk := f.read(4096):
                hash_algorithm.update(chunk)
        if correct_checksum == hash_algorithm.hexdigest():
            stored_checksums[basename] = correct_checksum
            print(f"Checksum of {basename} is correct, saving to {checksumfile}")
            with open(checksumfile, "w") as fp:
                json.dump(stored_checksums, fp, indent=4)
            return True
        else:
            print(f"Checksum of {basename} was INCORRECT: {hash_algorithm.hexdigest()}")
    return False


GENERATE_REFERENCE_CHECKSUMS = False
for url, checksum in models_urls:
    basename = os.path.join(MODEL_DIR,os.path.basename(url))

    # create list of urls and checksum    
    if GENERATE_REFERENCE_CHECKSUMS:
        if os.path.exists(basename):
            hash_algorithm = hashlib.sha256()
            with open(basename, "rb") as f:
                while chunk := f.read(4096):
                    hash_algorithm.update(chunk)
            print(f"['{url}','{hash_algorithm.hexdigest()}'],")
    else: #run verification
        if verifychecksum(basename, checksum, saved_checksums):
            continue
        else:
            print(f"Verify checksum of {basename} FAILED, downloading")
            if os.path.exists(basename):
                os.remove(basename)

        if basename in saved_checksums:
            del saved_checksums[basename]
            with open(checksumfile, "w") as fp:
                json.dump(saved_checksums, fp, indent=4)
        print(f"Downloading {url}")
        os.system(f"wget -P {MODEL_DIR} {url}")

        if not verifychecksum(basename, checksum, saved_checksums):
            print(f"Verify Checksum of {basename} FAILED")
            if os.path.exists(basename):
                os.remove(basename)
