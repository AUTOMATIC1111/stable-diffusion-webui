#!/usr/bin/env node

// simple nodejs script to test sdnext api

const fs = require('fs'); // eslint-disable-line no-undef
const process = require('process'); // eslint-disable-line no-undef

const sd_url = process.env.SDAPI_URL || 'http://127.0.0.1:7860';
const sd_username = process.env.SDAPI_USR;
const sd_password = process.env.SDAPI_PWD;
const sd_options = {
  // first pass
  prompt: 'city at night',
  negative_prompt: 'foggy, blurry',
  sampler_name: 'UniPC',
  seed: -1,
  steps: 20,
  batch_size: 1,
  n_iter: 1,
  cfg_scale: 6,
  width: 512,
  height: 512,
  // enable second pass
  enable_hr: true,
  // second pass: upscale
  hr_upscaler: 'SCUNet GAN',
  hr_scale: 2.0,
  // second pass: hires
  hr_force: true,
  hr_second_pass_steps: 20,
  hr_sampler_name: 'UniPC',
  denoising_strength: 0.5,
  // second pass: refiner
  refiner_steps: 5,
  refiner_start: 0.8,
  refiner_prompt: '',
  refiner_negative: '',
  // api return options
  save_images: false,
  send_images: true,
};

async function main() {
  const method = 'POST';
  const headers = new Headers();
  const body = JSON.stringify(sd_options);
  headers.set('Content-Type', 'application/json');
  if (sd_username && sd_password) headers.set({ Authorization: `Basic ${btoa('sd_username:sd_password')}` });
  const res = await fetch(`${sd_url}/sdapi/v1/txt2img`, { method, headers, body });
  if (res.status !== 200) {
    console.log('Error', res.status);
  } else {
    const json = await res.json();
    console.log('result:', json.info);
    for (const i in json.images) { // eslint-disable-line guard-for-in
      const f = `/tmp/test-{${i}.jpg`;
      fs.writeFileSync(f, atob(json.images[i]), 'binary');
      console.log('image saved:', f);
    }
  }
}

main();
