#!/usr/bin/env bash

DIR=(models/Stable-diffusion models/Lora models/VAE embeddings)
for D in ${DIR[@]}
do
  mkdir -p $HOME/stable-diffusion-webui/$D
  cd $HOME/stable-diffusion-webui/$D
  ln -sf /${VOLUME}/$D/* .
  cd -
done