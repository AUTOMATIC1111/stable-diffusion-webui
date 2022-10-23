import aioprocessing
import asyncio
import discord
import io
import json
import os
import queue
import random
import re
import threading
import time

from modules.paths import script_path

from modules import devices, sd_samplers, upscaler
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
import modules.processing as processing

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts

# labels for commands
PREFIX = "prefix"
NEG_PREFIX = "neg-prefix"
PROMPT = "prompt"
NEG_PROMPT = "neg-prompt"
STEPS = "steps"
CFG = "cfg"
SAMPLER = "sampler"
SEED = "seed"

SET_STEPS = "set-steps"
SET_CFG = "set-cfg"
SET_SAMPLER = "set-sampler"

KEYWORDS = [PREFIX, NEG_PREFIX, PROMPT, NEG_PROMPT, STEPS, CFG, SAMPLER, SEED, SET_STEPS, SET_CFG, SET_SAMPLER]

DEFAULT_STEPS = 28
MAX_STEPS = 100
DEFAULT_CFG = 11
MAX_CFG = 30
DEFAULT_SAMPLER = 1

BATCH_SIZE = 4

DEFAULT_TOKEN_GEN_RATE = 2

QUEUE_MAX_SIZE = 20

# work queue
work_queue = queue.Queue()

class WorkItem:
    def __init__(self, prompt, neg_prompt, steps, cfg, sampler, seed, message):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.seed = seed
        self.message = message
        self.event = aioprocessing.AioEvent()
    
    def get_prompt(self):
        return self.prompt
        
    def get_neg_prompt(self):
        return self.neg_prompt
    
    def get_steps(self):
        return self.steps
    
    def get_cfg(self):
        return self.cfg
    
    def get_sampler(self):
        return self.sampler

    def get_seed(self):
        return self.seed

    async def wait_complete(self):
        await self.event.coro_wait()

    def complete(self, pictures):
        self.pictures = pictures
        self.event.set()
    
    def get_pictures(self):
        return self.pictures
    
    def get_message(self):
        return self.message

# process stable diffusion work item
def process_sd_work(queued_call_lock, work_item):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        prompt=work_item.get_prompt(),
        negative_prompt=work_item.get_neg_prompt(),
        seed=work_item.get_seed(),
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_index=work_item.get_sampler(),
        batch_size=BATCH_SIZE,
        n_iter=1,
        steps=work_item.get_steps(),
        cfg_scale=work_item.get_cfg(),
        width=512,
        height=512,
        do_not_save_samples=True,
        do_not_save_grid=True,
    )

    with queued_call_lock:
        processed = process_images(p)

    work_item.complete(processed.images)

# worker to consume items off the queue
def sd_worker():
    queued_call_lock = threading.Lock()
    def wrap_queued_call(func):
        def f(*args, **kwargs):
            with queued_call_lock:
                res = func(*args, **kwargs)

            return res

        return f
        
    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts()

    modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    while True:
        work_item = work_queue.get()
        process_sd_work(queued_call_lock, work_item)

class UserPreferencesDict:
    def __init__(self, preferences = {}):
        self.preferences = preferences
        self.lock = threading.Lock()

    def get_preference(self, user, pref_name):
        user = str(user)
        with self.lock:
            if user in self.preferences and pref_name in self.preferences[user]:
                return self.preferences[user][pref_name]
        
        return None

    def set_preference(self, user, pref_name, value):
        user = str(user)
        with self.lock:
            if user not in self.preferences:
                self.preferences[user] = {}
            
            self.preferences[user][pref_name] = value

    def to_dict(self):
        with self.lock:
            return self.preferences

def load_preferences(path):
    try:
        with open(path) as f:
            return UserPreferencesDict(json.load(f))
    except FileNotFoundError:
        return UserPreferencesDict()

def save_preferences(preferences, path):
    with open(path, 'w') as f:
        json.dump(preferences.to_dict(), f)

class StableDiffusionClient(discord.Client):
    async def on_ready(self):
        self.last_request_times = {}
        print("Logged on to discord as", self.user)
    
    def set_preferences(self, preferences):
        self.preferences = preferences

    def set_config(self, config):
        self.config = config

    def parse_message(self, message):
        request = {}

        for keyword in KEYWORDS:
            match = re.search("^%s:([^\n]*)" % (keyword), message, re.MULTILINE)
            if match != None:
                request[keyword] = match.group(1).strip()
        
        return request

    async def send_response(self, work_item, channel):
        async with channel.typing():
            await work_item.wait_complete()
            pictures = work_item.get_pictures()
            message = work_item.get_message()

            files = []
            for picture in pictures:
                img = io.BytesIO()
                picture.save(img, format="png")
                img.seek(0)
                files.append(discord.File(img, filename="ai_image.png"))

            await message.reply(files=files)

    def get_token_rate(self, user : int) -> int:
        user = str(user)
        if user in self.config["token_rate"]:
            return self.config["token_rate"][user]
        
        return DEFAULT_TOKEN_GEN_RATE
            

    def time_since_last_request(self, user : int) -> int:
        if user in self.last_request_times:
            return time.time() - self.last_request_times[user]
        
        return 999
    
    def record_user_request(self, user : int) -> int:
        self.last_request_times[user] = time.time()

    def process_message(self, message):
        response = []
        request = self.parse_message(message.content)
        
        def ignore_exception(exception=Exception, default_val=None):
            def decorator(function):
                def wrapper(*args, **kwargs):
                    try:
                        return function(*args, **kwargs)
                    except exception:
                        return default_val
                return wrapper
            return decorator

        def sampler_string_to_idx(sampler_name):
            if sampler_name == "Euler a":
                return 0
            return 1

        def sampler_idx_to_string(sampler):
            if sampler == 0:
                return "Euler a"
            elif sampler == 1:
                return "Euler"
            
            return "Unknown"

        def bound_value(value, name):
            if value < 0:
                return 0
            
            if name == STEPS:
                if value > MAX_STEPS:
                    return MAX_STEPS
            elif name == CFG:
                if value > MAX_CFG:
                    return MAX_CFG
            else:
                print("unknown field type in bounding value")
                return 0

            return value

        # save preferences
        if PREFIX in request:
            response.append("Setting prompt prefix as: %s" % request[PREFIX])
            self.preferences.set_preference(message.author.id, PREFIX, request[PREFIX])
        if NEG_PREFIX in request:
            response.append("Setting negative prompt prefix as: %s" % request[NEG_PREFIX])
            self.preferences.set_preference(message.author.id, NEG_PREFIX, request[NEG_PREFIX])
        if SET_STEPS in request:
            steps = bound_value(ignore_exception(ValueError, DEFAULT_STEPS)(int)(request[SET_STEPS]), STEPS)
            response.append("Setting default steps to %d" % steps)
            self.preferences.set_preference(message.author.id, STEPS, steps)
        if SET_CFG in request:
            cfg = bound_value(ignore_exception(ValueError, DEFAULT_CFG)(float)(request[SET_CFG]), CFG)
            response.append("Setting default cfg to %d" % cfg)
            self.preferences.set_preference(message.author.id, CFG, cfg)
        if SET_SAMPLER in request:
            sampler = sampler_string_to_idx(request[SET_SAMPLER])
            response.append("Setting default sampler to %s" % sampler_idx_to_string(sampler))
            self.preferences.set_preference(message.author.id, SAMPLER, sampler)

        def make_prompt(request, name, prefix):
            if name in request:
                if prefix is not None:
                    return prefix + ", " + request[name]
                
                return request[name]
            
            return "" if prefix is None else prefix
        
        def make_param(request, name, func, default):
            if name in request:
                return func(request[name])
            
            preference = self.preferences.get_preference(message.author.id, name)
            if preference is not None:
                return preference

            return default

        if PROMPT in request or NEG_PROMPT in request:
            # don't make the queue too large
            if work_queue.qsize() > QUEUE_MAX_SIZE:
                return ["Work queue is at maximum size, please wait before making your next request"]
            
            prompt = make_prompt(request, PROMPT, self.preferences.get_preference(message.author.id, PREFIX))
            neg_prompt = make_prompt(request, NEG_PROMPT, self.preferences.get_preference(message.author.id, NEG_PREFIX))

            response.append("Generating %d images for prompt: %s" % (BATCH_SIZE, prompt))
            response.append("negative prompt: %s" % neg_prompt)

            steps = bound_value(make_param(request, STEPS, ignore_exception(ValueError, DEFAULT_STEPS)(int), DEFAULT_STEPS), STEPS)

            # make sure they have the tokens to make a request for this many steps
            user = message.author.id
            token_deficit = steps - self.time_since_last_request(user) * self.get_token_rate(user)
            if token_deficit > 0:
                return ["Please wait at least %d seconds before making your next request" % int(token_deficit/self.get_token_rate(user))]

            cfg = bound_value(make_param(request, CFG, ignore_exception(ValueError, DEFAULT_CFG)(float), DEFAULT_CFG), CFG)
            sampler = make_param(request, SAMPLER, sampler_string_to_idx, DEFAULT_SAMPLER)
            random_seed = int(random.randrange(4294967294))
            seed = make_param(request, SEED, ignore_exception(ValueError, random_seed)(int), random_seed)

            response.append("Using steps: %d, cfg: %.2f, sampler: %s, seed %d" % (steps, cfg, sampler_idx_to_string(sampler), seed))

            work_item = WorkItem(prompt, neg_prompt, steps, cfg, sampler, seed, message)
            work_queue.put(work_item)

            asyncio.get_event_loop().create_task(self.send_response(work_item, message.channel))
            self.record_user_request(user)
        
        return response

    async def on_message(self, message):
        channel_filter = self.config["channels"]

        # don't respond to ourselves
        if message.author == self.user:
            return
        
        # restrict to specific channels if specified
        if len(channel_filter) > 0 and message.channel.id not in channel_filter:
            return

        if message.content:
            response = self.process_message(message)
            if len(response) > 0:
                await message.channel.send("\n".join(response))

# worker to log into discord and create work items
def discord_worker(preferences, config):
    api_key = os.environ.get("DISCORD_API_KEY")
    if api_key is None:
        print("Please set DISCORD_API_KEY before use")
        return

    intents = discord.Intents.default()
    intents.message_content = True

    client = StableDiffusionClient(intents=intents)
    client.set_preferences(preferences)
    client.set_config(config)
    client.run(api_key)

def load_config(path):
    with open(path) as f:
        return json.load(f)

def main():
    preferences_file = "user_prefixes.json"
    config_file = "discord_config.json"

    # load user prefixes
    preferences = load_preferences(preferences_file)
    config = load_config(config_file)

    # start sd worker
    threading.Thread(target=sd_worker, daemon=True).start()

    # start discord bot
    threading.Thread(target=discord_worker, daemon=True, args=(preferences, config)).start()

    # run until termination, printing statistics
    try:
        while(True):
            print("Work queue depth %d" % (work_queue.qsize()))
            time.sleep(30)
            save_preferences(preferences, preferences_file)
    except KeyboardInterrupt:
        save_preferences(preferences, preferences_file)

if __name__ == "__main__":
    main()