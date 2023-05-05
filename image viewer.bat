TITLE image viewer
::todo: change the cd to absolute path
::cd outputs/txt2img-images && python -m http.server 8081
cd . && python http_gallery.py 8081 --directory outputs/txt2img-images