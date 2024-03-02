

# Generate a well known ssl cert CA store to be trusted
https://pypi.org/project/certifi/

pip install certifi

python -m certifi


# Check venv ssl cert trust settings
https://stackoverflow.com/questions/39356413/how-to-add-a-custom-ca-root-certificate-to-the-ca-store-used-by-pip-in-windows
python -c "import ssl; print(ssl.get_default_verify_paths())"



# Use the generated truststore and move it to the location being used
cp /Users/samuelwang/sd/stable-diffusion-webui/venv/lib/python3.10/site-packages/certifi/cacert.pem /Library/Frameworks/Python.framework/Versions/3.10/etc/openssl/cert.pem



# Prompt Process
UI inputs provided by users
users click on the generate button 
call queue wrap inputs and invoke txt2img run process (call_queue.py)
the result is stored in an Processed instance
the json inputs are generated from the Processed instance

We do not want to wait until it is processed because we won't process images locally.
So we need to create a js() method pre process, aka, Processing


    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_name": self.sd_model_name,
            "sd_model_hash": self.sd_model_hash,
            "sd_vae_name": self.sd_vae_name,
            "sd_vae_hash": self.sd_vae_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
            "version": self.version,
        }

        return json.dumps(obj)