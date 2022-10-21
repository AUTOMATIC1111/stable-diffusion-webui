import csv, os, shutil
import modules.scripts as scripts
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.shared import opts
import gradio
class Script(scripts.Script):
    def title(self):
        return "Create inspiration images"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        file = gradio.Files(label="Artist or styles name list. '.txt' files with one name per line",)
        with gradio.Row():
            prefix = gradio.Textbox("a painting in", label="Prompt words before artist or style name", file_count="multiple")
            suffix= gradio.Textbox("style", label="Prompt words after artist or style name")
        negative_prompt = gradio.Textbox("picture frame, portrait photo", label="Negative Prompt")
        with gradio.Row():
            batch_size = gradio.Number(1, label="Batch size")
            batch_count = gradio.Number(2, label="Batch count")            
        return [batch_size, batch_count, prefix, suffix, negative_prompt, file]

    def run(self, p, batch_size, batch_count, prefix, suffix, negative_prompt, files):
        p.batch_size = int(batch_size)
        p.n_iterint = int(batch_count)
        p.negative_prompt = negative_prompt        
        p.do_not_save_samples = True
        p.do_not_save_grid = True        
        for file in files:
            tp = file.orig_name.split(".")[0]
            print(tp)
            path = os.path.join(opts.inspiration_dir, tp)
            if not os.path.exists(path):
                os.makedirs(path) 
            f = open(file.name, "r")
            line = f.readline() 
            while len(line) > 0: 
                name = line.rstrip("\n").split(",")[0]
                line = f.readline() 
                artist_path = os.path.join(path, name)  
                if not os.path.exists(artist_path):
                    os.mkdir(artist_path)  
                if len(os.listdir(artist_path)) >= opts.inspiration_max_samples:
                    continue
                p.prompt = f"{prefix} {name} {suffix}"
                print(p.prompt)
                processed = processing.process_images(p)
                for img in  processed.images:
                    i = 0
                    filename = os.path.join(artist_path,  format(0, "03d") + ".jpg")
                    while os.path.exists(filename):
                        i += 1
                        filename = os.path.join(artist_path,  format(i, "03d") + ".jpg")
                    img.save(filename, quality=80)
        return processed
