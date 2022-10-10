import os
import sys
import traceback

import torch
import tqdm
import html
import datetime

from PIL import Image,PngImagePlugin
from ..images import captionImageOverlay
import numpy as np
import base64
import json
import zlib

from modules import shared, devices, sd_hijack, processing, sd_models
import modules.textual_inversion.dataset

class EmbeddingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'TORCHTENSOR':obj.cpu().detach().numpy().tolist()}
        return json.JSONEncoder.default(self, obj)

class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, d):
        if 'TORCHTENSOR' in d:
            return torch.from_numpy(np.array(d['TORCHTENSOR']))
        return d

def embeddingToB64(data):
    d = json.dumps(data,cls=EmbeddingEncoder)
    return base64.b64encode(d.encode())

def embeddingFromB64(data):
    d = base64.b64decode(data)
    return json.loads(d,cls=EmbeddingDecoder)

def xorBlock(block):
    return np.bitwise_xor(block.astype(np.uint8),
                          ((np.random.RandomState(0xDEADBEEF).random(block.shape)*255).astype(np.uint8)) & 0x0F )

def styleBlock(block,sequence):
    im = Image.new('RGB',(block.shape[1],block.shape[0]))
    draw = ImageDraw.Draw(im)
    i=0
    for x in range(-6,im.size[0],8):
        for yi,y in enumerate(range(-6,im.size[1],8)):
            offset=0
            if yi%2==0:
                offset=4
            shade = sequence[i%len(sequence)]
            i+=1
            draw.ellipse((x+offset, y, x+6+offset, y+6), fill =(shade,shade,shade) )

    fg = np.array(im).astype(np.uint8) & 0xF0
    return block ^ fg

def insertImageDataEmbed(image,data):
    d = 3
    data_compressed = zlib.compress( json.dumps(data,cls=EmbeddingEncoder).encode(),level=9)
    dnp = np.frombuffer(data_compressed,np.uint8).copy()
    dnphigh = dnp >> 4
    dnplow  = dnp & 0x0F
    
    h = image.size[1]
    next_size = dnplow.shape[0] + (h-(dnplow.shape[0]%h))
    next_size = next_size + ((h*d)-(next_size%(h*d)))

    dnplow.resize(next_size)
    dnplow = dnplow.reshape((h,-1,d))

    dnphigh.resize(next_size)
    dnphigh = dnphigh.reshape((h,-1,d))

    edgeStyleWeights = list(data['string_to_param'].values())[0].cpu().detach().numpy().tolist()[0][:1024]
    edgeStyleWeights = (np.abs(edgeStyleWeights)/np.max(np.abs(edgeStyleWeights))*255).astype(np.uint8)

    dnplow   = styleBlock(dnplow,sequence=edgeStyleWeights)
    dnplow   = xorBlock(dnplow)    
    dnphigh  = styleBlock(dnphigh,sequence=edgeStyleWeights[::-1])
    dnphigh  = xorBlock(dnphigh)

    imlow  = Image.fromarray(dnplow,mode='RGB')
    imhigh = Image.fromarray(dnphigh,mode='RGB')

    background = Image.new('RGB',(image.size[0]+imlow.size[0]+imhigh.size[0]+2,image.size[1]),(0,0,0))
    background.paste(imlow,(0,0))
    background.paste(image,(imlow.size[0]+1,0))
    background.paste(imhigh,(imlow.size[0]+1+image.size[0]+1,0))

    return background

def crop_black(img,tol=0):
    mask = (img>tol).all(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),mask.shape[1]-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),mask.shape[0]-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def extractImageDataEmbed(image):
    d=3
    outarr = crop_black(np.array(image.getdata()).reshape(image.size[1],image.size[0],d ).astype(np.uint8) ) & 0x0F
    blackCols = np.where( np.sum(outarr, axis=(0,2))==0)
    if blackCols[0].shape[0] < 2:
        print('No Image data blocks found.')
        return None

    dataBlocklower = outarr[:,:blackCols[0].min(),:].astype(np.uint8)
    dataBlockupper = outarr[:,blackCols[0].max()+1:,:].astype(np.uint8)

    dataBlocklower = xorBlock(dataBlocklower)
    dataBlockupper = xorBlock(dataBlockupper)
    
    dataBlock = (dataBlockupper << 4) | (dataBlocklower)
    dataBlock = dataBlock.flatten().tobytes()   
    data = zlib.decompress(dataBlock)
    return json.loads(data,cls=EmbeddingDecoder)

class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class EmbeddingDatabase:
    def __init__(self, embeddings_dir):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.dir_mtime = None
        self.embeddings_dir = embeddings_dir

    def register_embedding(self, embedding, model):

        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def load_textual_inversion_embeddings(self):
        mt = os.path.getmtime(self.embeddings_dir)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = []

            if filename.upper().endswith('.PNG'):
                embed_image = Image.open(path)
                if 'sd-ti-embedding' in embed_image.text:
                    data = embeddingFromB64(embed_image.text['sd-ti-embedding'])
                    name = data.get('name',name)
                else:
                    data = extractImageDataEmbed(embed_image)
                    name = data.get('name',name)
            else:
                data = torch.load(path, map_location="cpu")

            # textual inversion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

            vec = emb.detach().to(devices.device, dtype=torch.float32)
            embedding = Embedding(vec, name)
            embedding.step = data.get('step', None)
            embedding.sd_checkpoint = data.get('hash', None)
            embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
            self.register_embedding(embedding, shared.sd_model)

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                process_file(fullfn, fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} textual inversion embeddings.")

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, init_text='*'):
    cond_model = shared.sd_model.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    fn = os.path.join(shared.cmd_opts.embeddings_dir, f"{name}.pt")
    assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


def train_embedding(embedding_name, learn_rate, data_root, log_directory, training_width, training_height, steps, num_repeats, create_image_every, save_embedding_every, template_file, save_image_with_stored_embedding):
    assert embedding_name, 'embedding not selected'

    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None
        
    cond_model = shared.sd_model.cond_stage_model

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=num_repeats, placeholder_token=embedding_name, model=shared.sd_model, device=devices.device, template_file=template_file)

    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    embedding.vec.requires_grad = True

    optimizer = torch.optim.AdamW([embedding.vec], lr=learn_rate)

    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"

    ititial_step = embedding.step or 0
    if ititial_step > steps:
        return embedding, filename

    tr_img_len = len([os.path.join(data_root, file_path) for file_path in os.listdir(data_root)])
    epoch_len = (tr_img_len * num_repeats) + tr_img_len

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, (x, text) in pbar:
        embedding.step = i + ititial_step

        if embedding.step > steps:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = cond_model([text])

            x = x.to(devices.device)
            loss = shared.sd_model(x.unsqueeze(0), c)[0]
            del x

            losses[embedding.step % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_num = embedding.step // epoch_len
        epoch_step = embedding.step - (epoch_num * epoch_len) + 1

        pbar.set_description(f"[Epoch {epoch_num}: {epoch_step}/{epoch_len}]loss: {losses.mean():.7f}")

        if embedding.step > 0 and embedding_dir is not None and embedding.step % save_embedding_every == 0:
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-{embedding.step}.pt')
            embedding.save(last_saved_file)

        if embedding.step > 0 and images_dir is not None and embedding.step % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{embedding_name}-{embedding.step}.png')

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=text,
                steps=20,
				height=training_height,
				width=training_width,
                do_not_save_grid=True,
                do_not_save_samples=True,
            )

            processed = processing.process_images(p)
            image = processed.images[0]

            shared.state.current_image = image

            if save_image_with_stored_embedding and os.path.exists(last_saved_file):
                
                last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{embedding.step}.png')

                info = PngImagePlugin.PngInfo()
                data = torch.load(last_saved_file)
                info.add_text("sd-ti-embedding", embeddingToB64(data))

                title = "<{}>".format(data.get('name','???'))
                checkpoint = sd_models.select_checkpoint()
                footer_left = checkpoint.model_name
                footer_mid = '[{}]'.format(checkpoint.hash)
                footer_right = '{}'.format(embedding.step)

                captioned_image = captionImageOverlay(image,title,footer_left,footer_mid,footer_right)
                captioned_image = insertImageDataEmbed(captioned_image,data)

                captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
            
            image.save(last_saved_image)

            last_saved_image += f", prompt: {text}"

        shared.state.job_no = embedding.step

        shared.state.textinfo = f"""
<p>
Loss: {losses.mean():.7f}<br/>
Step: {embedding.step}<br/>
Last prompt: {html.escape(text)}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    checkpoint = sd_models.select_checkpoint()

    embedding.sd_checkpoint = checkpoint.hash
    embedding.sd_checkpoint_name = checkpoint.model_name
    embedding.cached_checksum = None
    embedding.save(filename)

    return embedding, filename

