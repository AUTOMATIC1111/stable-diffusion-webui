import torch
import sys
sys.path.insert(0,'./facelib/detection/yolov5face')
model = torch.load('facelib/detection/yolov5face/yolov5n-face.pt', map_location='cpu')['model']
torch.save(model.state_dict(),'weights/facelib/yolov5n-face.pth')