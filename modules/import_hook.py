import sys

# solution to load xformers by @liuzimo https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/6594#discussioncomment-5216054
# sd_hijack will fail to load xformers because sys.modules cannot find i. This solution solves the problem. 
if "--xformers" not in "".join(sys.argv):
    import xformers
sys.modules["xformers"] = xformers
