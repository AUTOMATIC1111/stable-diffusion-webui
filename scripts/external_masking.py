import math
import os
import sys
import traceback


import cv2
from PIL import Image
import numpy as np

lastx,lasty=None,None
zoomOrigin = 0,0
zoomFactor = 1

midDragStart = None

def display_mask_ui(image,mask,max_size,initPolys):
  global lastx,lasty,zoomOrigin,zoomFactor

  lastx,lasty=None,None
  zoomOrigin = 0,0
  zoomFactor = 1

  polys = initPolys

  def on_mouse(event, x, y, buttons, param):
    global lastx,lasty,zoomFactor,midDragStart,zoomOrigin

    lastx,lasty = (x+zoomOrigin[0])/zoomFactor,(y+zoomOrigin[1])/zoomFactor

    if event == cv2.EVENT_LBUTTONDOWN:
      polys[-1].append((lastx,lasty))
    elif event == cv2.EVENT_RBUTTONDOWN:
      polys.append([])
    elif event == cv2.EVENT_MBUTTONDOWN:
        midDragStart = zoomOrigin[0]+x,zoomOrigin[1]+y
    elif event == cv2.EVENT_MBUTTONUP:
        if midDragStart is not None:
            zoomOrigin = max(0,midDragStart[0]-x),max(0,midDragStart[1]-y)
        midDragStart = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if midDragStart is not None:
            zoomOrigin = max(0,midDragStart[0]-x),max(0,midDragStart[1]-y)
    elif event == cv2.EVENT_MOUSEWHEEL:
        origZoom = zoomFactor
        if buttons > 0:
            zoomFactor *= 1.1
        else:
            zoomFactor *= 0.9
        zoomFactor = max(1,zoomFactor)

        zoomOrigin = max(0,int(zoomOrigin[0]+ (max_size*0.25*(zoomFactor-origZoom)))) , max(0,int(zoomOrigin[1] + (max_size*0.25*(zoomFactor-origZoom))))



  opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

  if mask is None:
    opencvMask  = cv2.cvtColor( np.array(opencvImage) , cv2.COLOR_BGR2GRAY)
  else:
    opencvMask  = np.array(mask)


  maxdim = max(opencvImage.shape[1],opencvImage.shape[0])

  factor = max_size/maxdim


  cv2.namedWindow('MaskingWindow', cv2.WINDOW_AUTOSIZE)
  cv2.setWindowProperty('MaskingWindow', cv2.WND_PROP_TOPMOST, 1)
  cv2.setMouseCallback('MaskingWindow', on_mouse)

  font = cv2.FONT_HERSHEY_SIMPLEX

  srcImage = opencvImage.copy()
  combinedImage = opencvImage.copy()

  interp = cv2.INTER_CUBIC
  if zoomFactor*factor < 0:
    interp = cv2.INTER_AREA

  zoomedSrc = cv2.resize(srcImage,(None,None),fx=zoomFactor*factor,fy=zoomFactor*factor,interpolation=interp)
  zoomedSrc = zoomedSrc[zoomOrigin[1]:zoomOrigin[1]+max_size,zoomOrigin[0]:zoomOrigin[0]+max_size,:]

  lastZoomFactor = zoomFactor
  lastZoomOrigin = zoomOrigin
  while 1:

    if lastZoomFactor != zoomFactor or lastZoomOrigin != zoomOrigin:
        interp = cv2.INTER_CUBIC
        if zoomFactor*factor < 0:
          interp = cv2.INTER_AREA
        zoomedSrc = cv2.resize(srcImage,(None,None),fx=zoomFactor*factor,fy=zoomFactor*factor,interpolation=interp)
        zoomedSrc = zoomedSrc[zoomOrigin[1]:zoomOrigin[1]+max_size,zoomOrigin[0]:zoomOrigin[0]+max_size,:]
        zoomedSrc = cv2.copyMakeBorder(zoomedSrc, 0, max_size-zoomedSrc.shape[0], 0, max_size-zoomedSrc.shape[1], cv2.BORDER_CONSTANT)

        lastZoomFactor = zoomFactor
        lastZoomOrigin = zoomOrigin

    foreground    = np.zeros_like(zoomedSrc)

    for i,polyline in enumerate(polys):
      if len(polyline)>0:

        segs = polyline[::]

        active=False
        if len(polys[-1])>0 and i==len(polys)-1 and lastx is not None:
          segs = polyline+[(lastx,lasty)]
          active=True

        segs = np.array(segs) - np.array([(zoomOrigin[0]/zoomFactor,zoomOrigin[1]/zoomFactor)])
        segs = (np.array([segs])*zoomFactor).astype(int)
        
        if active:
          cv2.fillPoly(foreground, (np.array(segs)) , ( 190, 107,  253), 0)
        else:
          cv2.fillPoly(foreground, (np.array(segs)) , (255, 255, 255), 0)

        if active:
            for x,y in segs[0]:
                cv2.circle(foreground, (int(x),int(y)), 5, (25,25,25), 3)
                cv2.circle(foreground, (int(x),int(y)), 5, (255,255,255), 2)


    foreground[foreground<1] = zoomedSrc[foreground<1]
    combinedImage = cv2.addWeighted(zoomedSrc, 0.5, foreground, 0.5, 0)

    helpText='Q=Save, C=Reset, LeftClick=Add new point to polygon, Rightclick=Close polygon, MouseWheel=Zoom, MidDrag=Pan'
    combinedImage = cv2.putText(combinedImage, helpText, (0,11), font, 0.4, (0,0,0), 2, cv2.LINE_AA)
    combinedImage = cv2.putText(combinedImage, helpText, (0,11), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('MaskingWindow',combinedImage)

    try:
      key = cv2.waitKey(1)
      if key == ord('q'):
        if len(polys[0])>0:
          newmask = np.zeros_like(cv2.cvtColor( opencvMask.astype('uint8') ,cv2.COLOR_GRAY2BGR) )
          for i,polyline in enumerate(polys):
            if len(polyline)>0:
              segs = [(int(a/factor),int(b/factor)) for a,b in polyline]
              cv2.fillPoly(newmask, np.array([segs]), (255,255,255), 0)
          cv2.destroyWindow('MaskingWindow')
          return Image.fromarray( cv2.cvtColor( newmask, cv2.COLOR_BGR2GRAY) ),polys
        break
      if key == ord('c'):
        polys = [[]]

    except Exception as e:
      print(e)
      break

  cv2.destroyWindow('MaskingWindow')
  return mask,polys

if __name__ == '__main__':
    img  = Image.open('K:\\test2.png')
    oldmask = Image.new('L',img.size,(0,))
    newmask,newPolys = display_mask_ui(img,oldmask,1024,[[]])

    opencvImg  = cv2.cvtColor( np.array(img) , cv2.COLOR_RGB2BGR)
    opencvMask  = cv2.cvtColor( np.array(newmask) , cv2.COLOR_GRAY2BGR)

    combinedImage = cv2.addWeighted(opencvImg, 0.5, opencvMask, 0.5, 0)
    combinedImage = Image.fromarray( cv2.cvtColor(  combinedImage  , cv2.COLOR_BGR2RGB))
    
    display_mask_ui(combinedImage,oldmask,1024,[[]])


    exit()

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):

    def title(self):
        return "External Image Masking"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        initialSize = 1024

        try:
          import tkinter as tk
          root = tk.Tk()
          screen_width  = int(root.winfo_screenwidth())
          screen_height = int(root.winfo_screenheight())
          print(screen_width,screen_height)
          initialSize = min(screen_width,screen_height)-50
          print(initialSize)
        except Exception as e:
          print(e)

        max_size = gr.Slider(label="Masking preview size", minimum=512, maximum=initialSize*2, step=8, value=initialSize)
        with gr.Row():
          ask_on_each_run      = gr.Checkbox(label='Draw new mask on every run', value=False)
          non_contigious_split = gr.Checkbox(label='Process non-contigious masks separately', value=False)

        return [max_size,ask_on_each_run,non_contigious_split]

    def run(self, p, max_size, ask_on_each_run, non_contigious_split):

        if not hasattr(self,'lastImg'):
          self.lastImg = None

        if not hasattr(self,'lastMask'):
          self.lastMask = None

        if not hasattr(self,'lastPolys'):
          self.lastPolys = [[]]

        if ask_on_each_run or self.lastImg is None or self.lastImg != p.init_images[0]:

          if self.lastImg is None or self.lastImg != p.init_images[0]:
            self.lastPolys = [[]]

          p.image_mask,self.lastPolys  = display_mask_ui(p.init_images[0],p.image_mask,max_size,self.lastPolys)
          self.lastImg  = p.init_images[0]
          if p.image_mask is not None:
            self.lastMask = p.image_mask.copy()
        elif hasattr(self,'lastMask') and self.lastMask is not None:
          p.image_mask = self.lastMask.copy()

        if non_contigious_split:
          maskImgArr = np.array(p.image_mask)
          ret, markers = cv2.connectedComponents(maskImgArr)
          markerCount = markers.max()

          if markerCount > 1:
            tempimages = []
            tempMasks  = []
            for maski in range(1,markerCount+1):
              print('maski',maski)
              maskSection = np.zeros_like(maskImgArr)
              maskSection[markers==maski] = 255
              p.image_mask = Image.fromarray( maskSection.copy() )
              proc = process_images(p)
              images = proc.images
              tempimages.append(np.array(images[0]))
              tempMasks.append(np.array(maskSection.copy()))

            finalImage = tempimages[0].copy()

            for outimg,outmask in zip(tempimages,tempMasks):

              resizeimg = cv2.resize(outimg, (finalImage.shape[0],finalImage.shape[1]) )
              resizedMask = cv2.resize(outmask, (finalImage.shape[0],finalImage.shape[1]) )
              
              finalImage[resizedMask==255] = resizeimg[resizedMask==255]
            images = [finalImage]


          else:
            proc = process_images(p)
            images = proc.images
        else:
          proc = process_images(p)
          images = proc.images

        proc.images = images
        return proc
