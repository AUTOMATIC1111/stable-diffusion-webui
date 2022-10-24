#!/bin/bash -e

bucket_name="my-bucket" # Change this to your own bucket
bucket_region="eu-west-1" # Change this to your bucket region

echo "Downloading models..."
# Add your steps to download the models. E.g.
aws --region ${bucket_region} s3 sync s3://${bucket_name}/models models/

echo "Downloading embeddings..."
mkdir -p embeddings
# Add your steps to download the embeddings. E.g.
aws --region ${bucket_region} s3 sync s3://${bucket_name}/embeddings embeddings/

export PYTHONPATH="${PYTHONPATH}:/home/sd/stable-diffusion-webui/repositories/stable-diffusion"

# Full list of params here: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py
# --no-half requires more than 10GB of GPU vRAM to work
python3 webui.py --listen --port 8080 --no-half