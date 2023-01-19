# learning notes

## Using accumulation: 1

- steps: 4500
- batch': 2
- accumulation: 1
- rate: "0.005000:100, 0.002500:300, 0.001000:600, 0.000500:1000, 0.000250:1500, 0.000100:2100, 0.000050:2800, 0.000025:3600, 0.000010:4500",

## Using accumulation: 10

- steps: 200
- batch': 2
- accumulation: 10
- learning-rate: '0.010:10, 0.008:20, 0.006:40, 0.004:80, 0.002:120, 0.001:160, 0.0005:200'

## Train

    sdapi.py interrupt
    rm -rf /tmp/train/ ~/dev/automatic/embeddings/rebeccagivens-v6.pt ~/dev/automatic/train/log/rebeccagivens-v6*
    train.py --name rebeccagivens-v6 --src ~/generative/Input/rebeccagivens/ --init person,woman,girl,model --overwrite

## Prompt

    a medium shot photo of "dreamkelly", extremely detailed 8k wallpaper, intricate, high detail, dramatic, modelshoot style


## Gen

> abby  ana  buffytyler  cassie  dreamkelly  hanna  katy  laurentaylor  lee  leila  linsiyee  marcel  mia  rebeccagivens  sahar  video  vlado

train.py --batch 2 --name ana --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name abby --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name buffytyler --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name cassie --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name dreamkelly --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name hanna --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name katy --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name laurentaylor --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name leila --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name linsiyee --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name mia --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name rebeccagivens --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name sahar --src ~/generative/Input/ana/ --init person,woman,girl,model
train.py --batch 2 --name lee --src ~/generative/Input/ana/ --init person,man
train.py --batch 2 --name marcel --src ~/generative/Input/ana/ --init person,man
train.py --batch 2 --name vlado --src ~/generative/Input/ana/ --init person,man
