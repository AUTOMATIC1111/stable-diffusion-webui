import os
import sys
import traceback

import torch
import tqdm
import html
import datetime

from modules import shared, devices, sd_hijack, processing
import modules.textual_inversion.dataset


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
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
        self.ids_lookup[first_id].append((ids, embedding))

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


def train_embedding(embedding_name, learn_rate, data_root, log_directory, steps, create_image_every, save_embedding_every, template_file):
    assert embedding_name, 'embedding not selected'

    shared.state.textinfo = "Initializing textual inversion training..."
    shared.state.job_count = steps

    filename = os.path.join(shared.cmd_opts.embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%d-%m"), embedding_name)

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

    cond_model = shared.sd_model.cond_stage_model

    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, size=512, placeholder_token=embedding_name, model=shared.sd_model, device=devices.device, template_file=template_file)

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

    pbar = tqdm.tqdm(enumerate(ds), total=steps-ititial_step)
    for i, (x, text) in pbar:
        embedding.step = i + ititial_step

        if embedding.step > steps:
            break

        if shared.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = cond_model([text])
            loss = shared.sd_model(x.unsqueeze(0), c)[0]

            losses[embedding.step % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f"loss: {losses.mean():.7f}")

        if embedding.step > 0 and embedding_dir is not None and embedding.step % save_embedding_every == 0:
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-{embedding.step}.pt')
            embedding.save(last_saved_file)

        if embedding.step > 0 and images_dir is not None and embedding.step % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{embedding_name}-{embedding.step}.png')

            p = processing.StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                prompt=text,
                steps=20,
                do_not_save_grid=True,
                do_not_save_samples=True,
            )

            processed = processing.process_images(p)
            image = processed.images[0]

            shared.state.current_image = image
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

    embedding.cached_checksum = None
    embedding.save(filename)

    return embedding, filename

