# Taiyi stable-diffusion-webui
Stable Diffusion web UI for Taiyi


## step 1

Since Taiyi's text_encoder has been modified (BertModel vs CLIPTextModel), and webui currently only supports stable diffusion in English, it is necessary to use the webui project modified by Fengshenbang's own fork.

```
git clone git@github.com:IDEA-CCNL/stable-diffusion-webui.git
cd stable-diffusion-webui
```

## step 2

Run webui's own commands to check and install the environment, webui will pull down the required repositories in the stable-diffusion-webui/repositories directory, this process will take some time.

```
python launch.py
```

![image](https://user-images.githubusercontent.com/4384420/201310784-19f9032a-4b14-4dfe-9d46-8b353bc74c5b.png)

now error will appear.


```
No checkpoints found. When searching for checkpoints, looked at:
 - file /xxx/stable-diffusion-webui/model.ckpt
 - directory /xxx/stable-diffusion-webui/models/Stable-diffusion
Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.
```

## step 3 

Download Taiyi ckpt file from https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
place your **local_path**

(You can use the https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/convert_diffusers_to_original_stable_diffusion.py file to transfer your model to ckpt if you finetune our model.)

![image](https://user-images.githubusercontent.com/4384420/201311084-751b440b-1a08-41fd-9870-7e7aec9aff16.png)

Replace the content of stable-diffusion-webui/repositories/stable-diffusion/configs/stable-diffusion/v1-inference.yaml with our file  stable-diffusion-webui/repositories/stable-diffusion-taiyi/configs/stable-diffusion/v1-inference.yaml , and change the  version with your **local_path**

Replace the content of stable-diffusion-webui/repositories/stable-diffusionldm/modules/encoders/modules.py with our file stable-diffusion-webui/repositories/stable-diffusion-taiyi/ldm/modules/encoders/modules.py



## step 4

Run the pip install for the change of stable-diffusion repositories

```
cd repositories/stable-diffusion
pip install -e .
cd ../../
```

## step 5

Run the command to experience Taiyi-Stable-Diffusion-1B-Chinese-v0.1

```
python launch.py --ckpt local_path/Taiyi-Stable-Diffusion-1B-Chinese-v0.1.ckpt
```
