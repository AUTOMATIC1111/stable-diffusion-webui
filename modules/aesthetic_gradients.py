from functools import lru_cache
from collections import namedtuple
from typing import List
import torch
import torch.optim as optim
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
from modules import shared, modelloader


def adjust_for_aesthetic_gradient(conditioning, prompt):
    aesthetic_embedding = shared.opts.data['aesthetic_embedding']
    aesthetic_embedding_path = 'aesthetic_embeddings/' + aesthetic_embedding
    aesthetic_embedding_steps = shared.opts.data['aesthetic_embedding_steps']
    device = conditioning[0][0].cond.device
    
    return _adjust_for_aesthetic_gradient(device, prompt, aesthetic_embedding, aesthetic_embedding_path, aesthetic_embedding_steps)

@lru_cache(maxsize=128)
def _adjust_for_aesthetic_gradient(device, prompt, aesthetic_embedding, aesthetic_embedding_path, aesthetic_embedding_steps):
    aesthetic_lr = 0.0001
    aesthetic_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    print(f"Training clip model on aesthetic gradient embedding {aesthetic_embedding} with {aesthetic_embedding_steps} training steps")
    
    with torch.enable_grad():
        batch_encoding = aesthetic_tokenizer(
            prompt,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(device)

        # This is the model to be personalized
        aesthetic_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

        # We load the aesthetic embeddings
        image_embs = torch.load(aesthetic_embedding_path).to(device)

        # We compute the loss (similarity between the prompt embedding and the aesthetic embedding)
        image_embs /= image_embs.norm(dim=-1, keepdim=True)
        text_embs = aesthetic_clip_model.get_text_features(tokens)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        sim = text_embs @ image_embs.T
        loss = -sim

        # lr = 0.0001

        # We optimize the model to maximize the similarity
        optimizer = optim.Adam(aesthetic_clip_model.text_model.parameters(), aesthetic_lr)

        # T = 0
        for i in range(aesthetic_embedding_steps):
            optimizer.zero_grad()

            loss.mean().backward()
            optimizer.step()

            text_embs = aesthetic_clip_model.get_text_features(tokens)
            text_embs /= text_embs.norm(dim=-1, keepdim=True)
            sim = text_embs @ image_embs.T
            loss = -sim

        z = aesthetic_clip_model.text_model(input_ids=tokens).last_hidden_state

    print(f"Trained clip model on aesthetic gradient embedding {aesthetic_embedding} Loss:{loss.item()}")
    return z


def list_embeddings():
    shared.opts.data['aesthetic_embedding'] = None
    shared.opts.data['aesthetic_embedding_steps'] = 0

    
def get_embeddings():
    model_path = 'aesthetic_embeddings'
    embeddings = modelloader.load_models(model_path=model_path, ext_filter="*.pt")
  
    return [modelloader.friendly_name(file) + '.pt' for file in embeddings]
