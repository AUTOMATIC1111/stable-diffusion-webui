import csv, os, shutil
import modules.scripts as scripts
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed


class Script(scripts.Script):
    def title(self):
        return "Create artists style image"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
       return []
    def show(self, is_img2img):
        return not is_img2img

    def run(self, p): #, max_snapshoots_num):
        path = os.path.join("style_snapshoot", "artist")
        if not os.path.exists(path):
            os.makedirs(path) 
        p.do_not_save_samples = True
        p.do_not_save_grid = True
        p.negative_prompt = "portrait photo"
        f = open('artists.csv')
        f_csv = csv.reader(f)        
        for row in f_csv:
            name = row[0]
            artist_path = os.path.join(path, name)  
            if not os.path.exists(artist_path):
                os.mkdir(artist_path)  
            if len(os.listdir(artist_path)) > 0:
                continue
            print(name)
            p.prompt = name
            processed = processing.process_images(p)
            for img in  processed.images:
                i = 0
                filename = os.path.join(artist_path,  format(0, "03d") + ".jpg")
                while os.path.exists(filename):
                    i += 1
                    filename = os.path.join(artist_path,  format(i, "03d") + ".jpg")
                img.save(filename, quality=70)
        return processed
