## API guide by [@Kilvoctu](https://github.com/Kilvoctu)


I am not a programmer so my knowledge is very limited, but nonetheless after a lot of banging sticks together I was able to figure out how to use the API (small tangential gripe, highly technical and experienced people are not very good at helping beginners how to code). 

I'll go through everything I learned about how to interface with the API. Apologies if I get a lot of terms and such wrong:

-First, of course, is to run web ui with `--api` commandline argument
-This enables the api which can be reviewed at http://127.0.0.1:7860/docs (or whever the URL is + /docs)
The basic ones I'm interested in are these two. Let's just focus only on ` /sdapi/v1/txt2img`
![image](https://user-images.githubusercontent.com/2993060/198171114-ed1c5edd-76ce-4c34-ad73-04e388423162.png)

-When you expand that tab, it gives an example of a payload to send to the API. I used this often as reference.
![image](https://user-images.githubusercontent.com/2993060/198171454-5b826ded-5e73-4249-9c0c-a97b32c42569.png)

------

-So that's the backend. The API basically says what's available, what it's asking for, and where to send it. Now moving onto the frontend, I'll start with constructing a payload with the values I want. An example can be:
```
payload = {
    "prompt": "maltese puppy",
    "steps": 5
}
```
I can put in as few or as many values as I want in the payload. The API will use the defaults for anything I don't set.

-After that, I dump the json, whatever that means, then I can send it to the API
```
payload_json = json.dumps(payload)
response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', data=payload_json).json()
```
Again, this URL needs to match the web ui's URL.
If we execute this code, the web ui will generate an image based on the payload. That's great, but then what? There is no image anywhere...

------

-After the backend does its thing, the API sends the response back in a variable that was assigned above: `response`. The response contains three entries; "images", "parameters", and "info", and I have to find some way to get the information from these entries.
-"images" is the generated image, which is what I want mostly. There's no link or anything; it's a giant string of random characters, apparently we have to decode it. This is how I do it:
```
for i in response['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i)))
```
-With that, we have an image in the `image` variable that we can work with, for example saving it with `image.save('output.png')`.
-"parameters" shows what was sent to the API, which could be useful, but what I want in this case is "info". I use it to insert metadata into the image, so I can drop it into web ui PNG Info. For that I simply reference it with `response['info']`

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

payload_json = json.dumps(payload)
response = requests.post(url=f'{url}/sdapi/v1/txt2img', data=payload_json).json()

for i in response['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i)))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", str(response['info']))
    image.save('output.png', pnginfo=pnginfo)
```
-Import the things I need
-define the url and the payload to send
-send said payload to said url through the API
-when we get the response, grab "images" and decode it
-define a plugin to add png info, then add "info" into it
-at the end here, save the image with the png info

For a more complete implementation of a frontend, my Discord bot is [here](https://github.com/Kilvoctu/aiyabot) if anyone wants to look at it as an example. Most of the action happens in stablecog.py. There are many comments explaining what each code does. Also a reference can be https://github.com/Kilvoctu/aiyabot/pull/11, my PR where I added API support.


------

This guide can be found in [discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/3734) page.