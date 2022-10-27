# Author: Therefore Games
# https://github.com/ThereforeGames/txt2img2img

import os
import string

import modules.scripts as scripts
import gradio as gr
import math

from modules import images, shared
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state, Options
import modules.sd_samplers
import modules.img2img
import random

import torch
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

from txt2img2img.dependencies import shortcodes

base_dir = "./txt2img2img"
routines_dir = f"{base_dir}/routines"
prompt_templates_dir = f"{base_dir}/prompt_templates"

@shortcodes.register("choose", "/choose")
def handler(pargs, kwargs, context, content):
	parts = content.split('|')
	return random.choice(parts)

@shortcodes.register("eval", "/eval")
def handler(pargs, kwargs, context, content):
	return str(eval(content))

@shortcodes.register("file", "/file")
def handler(pargs, kwargs, context, content):
	with open(base_dir + "/" + content + ".txt", "r") as file:
		file_contents = file.read().replace('\n', ' ')
	return(shortcode_parser.parse(file_contents, context=None))

shortcode_parser = shortcodes.Parser(start="<", end=">", esc="\\\\")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_clip_token_for_string(test, string):
	batch_encoding = test.tokenizer(string, truncation=True, max_length=77, return_length=True,
		return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
	tokens = batch_encoding["input_ids"]
	count = torch.count_nonzero(tokens - 49407)
	return count.item()

def choose_string(my_string,delimiter = "|"):
	my_string = my_string.split(delimiter)
	return random.choice(my_string)

class Script(scripts.Script):
	def title(self):
		return "txt2img2img v1.0.0"

	def show(self, is_img2img):
		return not is_img2img

	def ui(self, is_img2img):
		if is_img2img:
			return None

		return []

	def run(self, p):
		def get_prompt_complexity(prompt): return get_clip_token_for_string(FrozenCLIPEmbedder().cuda(),prompt)

		def process_prompt_template(template_file):
			this_file = f"{prompt_templates_dir}/{template_file}.txt"
			try:
				with open(this_file, 'r') as file:
					template = file.read().replace('\n', '')
			except:
				template = template_file

			template = template.replace("$intro"," ".join(prompt_parts[0:img2img_term_index]))
			template = template.replace("$outro"," ".join(prompt_parts[img2img_term_index+1:len(prompt_parts)]))
			template = template.replace("$old_term",getattr(img_opts,'txt2img_term','subject'))
			template = template.replace("$new_term",getattr(img_opts,'img2img_term','subject'))
			template = template.replace("$prompt",p.prompt)
			template = template.replace("$denoising_strength",str(p.denoising_strength))
			template = template.replace("$cfg_scale",str(p.cfg_scale))
			template = template.replace("$complexity",str(get_prompt_complexity(p.prompt)))
			
			return(shortcode_parser.parse(template))

		img2img_term_index = -1

		samplers_dict = {}
		for i, sampler in enumerate(modules.sd_samplers.samplers):
			samplers_dict[sampler.name.lower()] = i

		modules.processing.fix_seed(p)

		img_opts = modules.shared.Options()	

		original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
		prompt_parts = original_prompt.split(" ")

		routine_files = os.listdir(routines_dir)
		routines = [x.split(".")[0] for x in routine_files]

		print(f"Found the following preset files: {routines}")

		for i, this_part in enumerate(prompt_parts):
			sanitized_part = this_part.translate(str.maketrans('', '', string.punctuation))

			if sanitized_part in routines:
				img2img_term_index = i
				
				# Load settings for this part
				img_opts.load(f"{routines_dir}/{sanitized_part}.json")
				prompt_method = getattr(img_opts,"prompt_method",0)

				# Enable support for randomized prompt paramters
				img_opts.txt2img_term = choose_string(img_opts.txt2img_term)
				img_opts.img2img_term = choose_string(img_opts.img2img_term)

				# Term replacement
				prompt_parts[i] = prompt_parts[i].replace(sanitized_part,img_opts.img2img_term)
				img_opts.img2img_prompt = " ".join(prompt_parts)
				prompt_parts[i] = prompt_parts[i].replace(img_opts.img2img_term, img_opts.txt2img_term)

				# TODO: Add support for use of multiple presets within a prompt. For now, let's just kill the loop
				break

		if img2img_term_index > -1:
			# Temporarily disable color correction for this, it tends to make my pics looks washed out
			if (getattr("img_opts","bypass_color_correction",True)):
				temp_color_correction = getattr("opts","img2img_color_correction",False)
				opts.img2img_color_correction = False

			# TODO: Remove duplicate code below
			denoising_original = getattr(img_opts,"denoising_strength",0.7)
			steps_original = getattr(img_opts,"steps",p.steps)
			p.denoising_strength = denoising_original
			cfg_scale_original = getattr(img_opts,"cfg_scale",p.cfg_scale)
			p.cfg_scale = cfg_scale_original
			p.steps = steps_original
			p.prompt = " ".join(prompt_parts)

			# First txt2img action here: 
			processed = process_images(p)
			processed_amt = len(processed.images)

			# Now use our newly created txt2img result(s) with img2img...
			chain_offset_idx = 0
			batch_offset_idx = 0
			final_cycle = False
			penultimate_cycle = False			
			while (True):
				if (getattr(img_opts,"daisychain",False) and not final_cycle):
					print(f"Loading daisychain config for further processing: {img_opts.img2img_term}.json")
					next_file = img_opts.img2img_term
					img_opts = modules.shared.Options()	 # TODO: Make sure this isn't leaking memory
					img_opts.load(f"{routines_dir}/{next_file}.json")

					# TODO: Remove duplicate code below
					prompt_parts[img2img_term_index] = getattr(img_opts,"txt2img_term","subject")
					if (not getattr(img_opts,"daisychain",False)): penultimate_cycle = True
					
					img_opts.img2img_prompt = " ".join(prompt_parts)
					prompt_method = getattr(img_opts,"prompt_method",0)
					denoising_original = getattr(img_opts,"denoising_strength",0.7)
					steps_original = getattr(img_opts,"steps",p.steps)
					p.denoising_strength = denoising_original
					cfg_scale_original = getattr(img_opts,"cfg_scale",p.cfg_scale)
					p.cfg_scale = cfg_scale_original
					p.steps = steps_original
				elif (chain_offset_idx == 0):
					final_cycle = True

				if (final_cycle):
					prompt_parts[img2img_term_index] = getattr(img_opts,"img2img_term","subject")
					img_opts.img2img_prompt = " ".join(prompt_parts)

				# setup for the magic AI that optimizes your settings
				if getattr(img_opts,"autoconfigure",False):
					from rembg import remove

					max_subject_size = getattr(img_opts,"max_subject_size",0.5)
					min_overfit = 1
					max_overfit = 10
					overfit = min(max_overfit,max(min_overfit,getattr(img_opts,"overfit",5)))

					# TODO: lol hardcoded btw
					min_cfg_scale = 4.0
					min_denoising_strength = 0.36

					total_pixels = p.width * p.height
							
				p.prompt = img_opts.img2img_prompt
				for i in range(processed_amt):
					this_idx = chain_offset_idx + i - batch_offset_idx
					if (getattr(img_opts,"autoconfigure",False)):
						# Determine best CFG and denoising settings based on size of the subject in the txt2img result				
						transparent_img = remove(processed.images[this_idx])
						# Uncomment the next line to see how the background detection works:
						# transparent_img.save("output.png")

						# Count number of transparent pixels
						transparent_pixels = 0
						img_data = transparent_img.load()
						for y in range(p.height):
							for x in range(p.width):
								pixel_data = img_data[x,y]
								if (pixel_data[3] <= 10): transparent_pixels += 1
						subject_size = 1 - transparent_pixels / total_pixels
						print(f"Detected {transparent_pixels} transparent pixel(s). Your subject is {subject_size*100}% of the canvas.")

						p.steps = steps_original + round(max(0,(subject_size - max_subject_size) / max_subject_size) * (p.steps / 2))
						print(f"Adjusted sampling steps to {p.steps}")

						# Calculate new values using sigmoid distribution, wow so fancy
						p.cfg_scale = max(min_cfg_scale,(p.cfg_scale * 2) * sigmoid(1 - (max_subject_size / subject_size) * (overfit / 5)))
						
						# Lower the denoising strength a bit based on how many more words come after our object
						p.denoising_strength = denoising_original - 0.01 * (get_prompt_complexity(p.prompt) - (img2img_term_index + 1)) - 0.02 * " ".join(prompt_parts[0:img2img_term_index+1]).count(')')
						
						# Apply another fancy distribution curve
						p.denoising_strength = max(min_denoising_strength,(p.denoising_strength * 2) * sigmoid(1 - (max_subject_size / subject_size) * (overfit/5)))

						print(f"Updated CFG scale to {p.cfg_scale} and denoising strength to {p.denoising_strength}")

					p.prompt = process_prompt_template(getattr(img_opts,"prompt_template","default"))
					
					p.init_images = processed.images[this_idx]
					p.sampler_index = samplers_dict.get(getattr(img_opts,"sampler_name","euler a").lower(), p.sampler_index)
					p.restore_faces = getattr(img_opts,"restore_faces",p.restore_faces)
					p.mask_mode = 0
					p.seed = getattr(img_opts,"seed",p.seed)
					p.negative_prompt = getattr(img_opts,"negative_prompt",p.negative_prompt)

					if p.seed == -1: p.seed = int(random.randrange(4294967294))

					# This feels a bit hacky, but it seems to work for now
					img2img_result = modules.img2img.img2img(
						p.prompt,
						p.negative_prompt,
						getattr(p,"prompt_style","None"),
						getattr(p,"prompt_style2","None"),
						p.init_images,
						None, # p.init_img_with_mask
						None, # p.init_mask
						p.mask_mode,
						p.steps,
						p.sampler_index,
						0, # p.mask_blur
						0, # p.inpainting_fill
						p.restore_faces,
						p.tiling,
						0, #p.switch_mode
						1,#p.batch_count
						1,#p.batch_size
						p.cfg_scale,
						p.denoising_strength,
						p.seed,
						p.subseed,
						p.subseed_strength,
						p.seed_resize_from_h,
						p.seed_resize_from_w,
						p.height,
						p.width,
						0, # p.resize_mode
						'Irrelevant', # p.sd_upscale_upscaler_name
						0, # p.sd_upscale_overlap
						True, # p.inpaint_full_res
						False, # p.inpainting_mask_invert
						0, # this is the *args tuple and I believe 0 indicates we are not using an extra script in img2img
					)

					# Get the image stored in the first index
					img2img_images = img2img_result[0]

					# Add the new image(s) to our main output
					processed.images.append(img2img_images[0])

				batch_offset_idx = processed_amt
				chain_offset_idx += processed_amt

				# Break out of indefinite loop when we're done daisychaining
				if (final_cycle): break
				elif (penultimate_cycle): final_cycle = True

			# revert color correction setting
			if (getattr("img_opts","bypass_color_correction",True)):
				opts.img2img_color_correction = temp_color_correction

			return (processed)

		return None