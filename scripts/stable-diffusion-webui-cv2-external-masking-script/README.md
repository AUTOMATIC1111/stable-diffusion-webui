## Provides an external cv2 powered masking tool for https://github.com/AUTOMATIC1111/stable-diffusion-webui

## Installation
**[Download the zipped script Here](https://github.com/dfaker/stable-diffusion-webui-cv2-external-masking-script/archive/refs/heads/main.zip)**
and copy the file external_masking.py into your scripts folder.

requires cv2 to be installed

`py -m pip install opencv-python`

## Guide

The UI inside stable-diffusion-webui is pretty simple 
![Screenshot 2022-09-16 091930](https://user-images.githubusercontent.com/35278260/190592056-644c59db-907d-4cf1-ba85-0014eceea12a.jpg)

`Masking preview size` controls the size of the popup CV2 window

`Draw new mask on every run` will popup a new window for a new mask each time generate is clicked, usually it'll only appear on the first run, or when the input image is changed.

The masking window itself is pretty minimal
![image](https://user-images.githubusercontent.com/35278260/193962552-3dfa4d28-5899-4e3f-a589-362de5990636.png)

Showing the polygon currently being drawn in pink, left clicking starts a new polygon, right clicking closes the current polycon being drawn.

C to the clear current mask.

Q to quit and pass the current mask back to stable-diffusion-webui

Scroll the mouse wheel to zoom in

Middle click and drag to pan around the image

The mask drawn with the script will not be shown on the input image, but will be used for all outputs:

![Screenshot 2022-09-16 091911](https://user-images.githubusercontent.com/35278260/190593109-10d47736-428c-4c3f-841a-a964778fbec7.jpg)
