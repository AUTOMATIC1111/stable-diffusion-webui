## API guide by [@Kilvoctu](https://github.com/Kilvoctu)

- First, of course, is to run web ui with `--api` commandline argument
  - example in your "webui-user.bat": `set COMMANDLINE_ARGS=--api`
- This enables the api which can be reviewed at http://127.0.0.1:7860/docs (or whever the URL is + /docs)
The basic ones I'm interested in are these two. Let's just focus only on ` /sdapi/v1/txt2img`

![image](https://user-images.githubusercontent.com/2993060/198171114-ed1c5edd-76ce-4c34-ad73-04e388423162.png)

- When you expand that tab, it gives an example of a payload to send to the API. I used this often as reference.

![image](https://user-images.githubusercontent.com/2993060/198171454-5b826ded-5e73-4249-9c0c-a97b32c42569.png)

------

- So that's the backend. The API basically says what's available, what it's asking for, and where to send it. Now moving onto the frontend, I'll start with constructing a payload with the values I want. An example can be:
```
payload = {
    "prompt": "maltese puppy",
    "steps": 5
}
```
I can put in as few or as many values as I want in the payload. The API will use the defaults for anything I don't set.

- After that, I can send it to the API
```
response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload)
```
Again, this URL needs to match the web ui's URL.
If we execute this code, the web ui will generate an image based on the payload. That's great, but then what? There is no image anywhere...

------

- After the backend does its thing, the API sends the response back in a variable that was assigned above: `response`. The response contains three entries; "images", "parameters", and "info", and I have to find some way to get the information from these entries.
- First, I put this line `r = response.json()` to make it easier to work with the response.
- "images" is the generated image, which is what I want mostly. There's no link or anything; it's a giant string of random characters, apparently we have to decode it. This is how I do it:
```
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
```
- With that, we have an image in the `image` variable that we can work with, for example saving it with `image.save('output.png')`.
- "parameters" shows what was sent to the API, which could be useful, but what I want in this case is "info". I use it to insert metadata into the image, so I can drop it into web ui PNG Info. For that, I can access the `/sdapi/v1/png-info` API. I'll need to feed the image I got above into it.
```
png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/png-info', json=png_payload)
```
After that, I can get the information with `response2.json().get("info")`

------

A sample code that should work can look like this:
```
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

This is as of commit [ac08562](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/ac085628540d0ec6a988fad93f5b8f2154209571)

For a more complete implementation of a frontend, my Discord bot is [here](https://github.com/Kilvoctu/aiyabot) if anyone wants to look at it as an example. Most of the action happens in stablecog.py. There are many comments explaining what each code does.

------

This guide can be found in [discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/3734) page.