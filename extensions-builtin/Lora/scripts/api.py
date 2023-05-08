from fastapi import FastAPI
import gradio as gr
import json
import os
import lora

def get_lora_prompts(path):
    directory, filename = os.path.split(path)
    name_without_ext = os.path.splitext(filename)[0]
    new_filename = name_without_ext + '.civitai.info'
    try:
        new_path = os.path.join(directory, new_filename)
        if os.path.exists(new_path):
            with open(new_path, 'r') as f:
                data = json.load(f)
                trained_words = data.get('trainedWords', [])
                if len(trained_words) > 0:
                    result = ','.join(trained_words)
                    return result
                else:
                    return ''
        else:
            return ''
    except Exception as e:
        return ''

def api(_: gr.Blocks, app: FastAPI):
    @app.get("/sdapi/v1/loras")
    async def get_loras():
        return [{"name": name, "path": lora.available_loras[name].filename, "prompt": get_lora_prompts(lora.available_loras[name].filename)} for name in lora.available_loras]

