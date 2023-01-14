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

> train.py --name laurentaylor-v4 --src ~/generative/Input/laurentaylor/ --init person,woman,girl,model --overwrite

## Prompt

    a medium shot photo of "dreamkelly", extremely detailed 8k wallpaper, intricate, high detail, dramatic, modelshoot style
