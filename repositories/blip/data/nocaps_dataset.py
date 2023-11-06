import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

class nocaps_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):   
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_test.json'}
        filenames = {'val':'nocaps_val.json','test':'nocaps_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):  
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        return image, int(ann['img_id'])    