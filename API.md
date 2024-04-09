## API guide by [@Kilvoctu](https://github.com/Kilvoctu)

- First, of course, is to run web ui with `--api` commandline argument
  - example in your "webui-user.bat": `set COMMANDLINE_ARGS=--api`
- This enables the api which can be reviewed at http://127.0.0.1:7860/docs (or whever the URL is + /docs)
The basic ones I'm interested in are these two. Let's just focus only on ` /sdapi/v1/txt2img`

![image](https://user-images.githubusercontent.com/2993060/198171114-ed1c5edd-76ce-4c34-ad73-04e388423162.png)

- When you expand that tab, it gives an example of a payload to send to the API. I used this often as reference.

![image](https://user-images.githubusercontent.com/2993060/198171454-5b826ded-5e73-4249-9c0c-a97b32c42569.png)

------

- So that's the backend. The API basically says what's available, what it's asking for, and where to send it. Now moving onto the frontend, I'll start with constructing a payload with the parameters I want. An example can be:
```py
payload = {
    "prompt": "maltese puppy",
    "steps": 5
}
```
I can put in as few or as many parameters as I want in the payload. The API will use the defaults for anything I don't set.

- After that, I can send it to the API
```py
response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload)
```
Again, this URL needs to match the web ui's URL.
If we execute this code, the web ui will generate an image based on the payload. That's great, but then what? There is no image anywhere...

------

- After the backend does its thing, the API sends the response back in a variable that was assigned above: `response`. The response contains three entries; "images", "parameters", and "info", and I have to find some way to get the information from these entries.
- First, I put this line `r = response.json()` to make it easier to work with the response.
- "images" is the generated image, which is what I want mostly. There's no link or anything; it's a giant string of random characters, apparently we have to decode it. This is how I do it:
```py
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
```
- With that, we have an image in the `image` variable that we can work with, for example saving it with `image.save('output.png')`.
- "parameters" shows what was sent to the API, which could be useful, but what I want in this case is "info". I use it to insert metadata into the image, so I can drop it into web ui PNG Info. For that, I can access the `/sdapi/v1/png-info` API. I'll need to feed the image I got above into it.
```py
png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/png-info', json=png_payload)
```
After that, I can get the information with `response2.json().get("info")`

------

A sample code that should work can look like this:
```py
import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

url = "http://127.0.0.1:7860"

payload = {
    "prompt": "puppy dog",
    "steps": 5
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

r = response.json()

for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output.png', pnginfo=pnginfo)
```
- Import the things I need
- define the url and the payload to send
- send said payload to said url through the API
- in a loop grab "images" and decode it
- for each image, send it to png info API and get that info back
- define a plugin to add png info, then add the png info I defined into it
- at the end here, save the image with the png info

-----

A note on `"override_settings"`.
The purpose of this endpoint is to override the web ui settings for a single request, such as the CLIP skip. The settings that can be passed into this parameter are visible here at the url's /docs.

![image](https://user-images.githubusercontent.com/2993060/202877368-c31a6e9e-0d05-40ec-ade0-49ed2c4be22b.png)

You can expand the tab and the API will provide a list. There are a few ways you can add this value to your payload, but this is how I do it. I'll demonstrate with "filter_nsfw", and "CLIP_stop_at_last_layers".

```py
payload = {
    "prompt": "cirno",
    "steps": 20
}

override_settings = {}
override_settings["filter_nsfw"] = true
override_settings["CLIP_stop_at_last_layers"] = 2

override_payload = {
                "override_settings": override_settings
            }
payload.update(override_payload)
```
- Have the normal payload
- after that, initialize a dictionary (I call it "override_settings", but maybe not the best name)
- then I can add as many key:value pairs as I want to it
- make a new payload with just this parameter
- update the original payload to add this one to it

So in this case, when I send the payload, I should get a "cirno" at 20 steps, with the CLIP skip at 2, as well as the NSFW filter on.


For certain settings or situations, you may want your changes to stay. For that you can post to the `/sdapi/v1/options` API endpoint
We can use what we learned so far and set up the code easily for this. Here is an example:
```py
url = "http://127.0.0.1:7860"

option_payload = {
    "sd_model_checkpoint": "Anything-V3.0-pruned.ckpt [2700c435]",
    "CLIP_stop_at_last_layers": 2
}

response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)
```
After sending this payload to the API, the model should swap to the one I set and set the CLIP skip to 2. Reiterating, this is different from "override_settings", because this change will persist, while "override_settings" is for a single request.
Note that if you're changing the `sd_model_checkpoint`, the value should be the name of the checkpoint as it appears in the web ui. This can be referenced with this API endpoint (same way we reference "options" API)

![image](https://user-images.githubusercontent.com/2993060/202928589-114aff91-2777-4269-9492-2eab015c5bca.png)

The "title" (name and hash) is what you want to use.

-----

This is as of commit [47a44c7](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/47a44c7e421b98ca07e92dbf88769b04c9e28f86)

For a more complete implementation of a frontend, my Discord bot is [here](https://github.com/Kilvoctu/aiyabot) if anyone wants to look at it as an example. Most of the action happens in stablecog.py. There are many comments explaining what each code does.

------

This guide can be found in [discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/3734) page.

Also, check out this python API client library for webui: https://github.com/mix1009/sdwebuiapi
Using custom scripts/extensions example: [here](https://github.com/mix1009/sdwebuiapi/commit/fe269dc2d4f8a98e96c63c8a7d3b5f039625bc18)