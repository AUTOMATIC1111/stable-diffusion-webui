## Example script
[txt2img and img2img example python script](https://gist.github.com/w-e-w/0f37c04c18e14e4ee1482df5c4eb9f53)
## Useful utility
[API payload display](https://github.com/huchenlei/sd-webui-api-payload-display) a extension that converts webui image generation call to JSON for API
###

## API guide by [@Kilvoctu](https://github.com/Kilvoctu)
> ℹ️ **Note:**
> As of 2023-09-09, this guide is currently not maintained, and information is likely out of date. Note that the internal docs can always be accessed via the `/docs` endpoint (i.e. http://127.0.0.1:7860/docs)

- First, of course, is to run webui with `--api` commandline argument
  - example in your "webui-user.bat": `set COMMANDLINE_ARGS=--api`
- This enables the API which can be reviewed at http://127.0.0.1:7860/docs (or whever the URL is + /docs)
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
  response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload)
  ```
  Again, this URL needs to match the web UI's URL.
  If we execute this code, the web UI will generate an image based on the payload. That's great, but then what? There is no image anywhere...

------

- After the backend does its thing, the API sends the response back in a variable that was assigned above: `response`. The response contains three entries; `images`, `parameters`, and `info`, and I have to find some way to get the information from these entries.
- First, I put this line `r = response.json()` to make it easier to work with the response.
- "images" is a list of base64-encoded generated images.
- From there, we can decode and save it.
  ```python
  with open("output.png", 'wb') as f:
      f.write(base64.b64decode(r['images'][0]))
  ```

A sample script that should work can look like this:
```py
import requests
import base64

# Define the URL and the payload to send.
url = "http://127.0.0.1:7860"

payload = {
    "prompt": "puppy dog",
    "steps": 5
}

# Send said payload to said URL through the API.
response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
r = response.json()

# Decode and save the image.
with open("output.png", 'wb') as f:
    f.write(base64.b64decode(r['images'][0]))
```
-----
A note on `override_settings`.
The purpose of this parameter is to override the webui settings such as model or CLIP skip for a single request. The settings that can be passed into this parameter are can be found at the URL `/docs`.

![image](https://user-images.githubusercontent.com/2993060/202877368-c31a6e9e-0d05-40ec-ade0-49ed2c4be22b.png)

You can expand the tab and the API will provide a list. There are a few ways you can add this value to your payload, but this is how I do it. I'll demonstrate with "sd_model_checkpoint", and "CLIP_stop_at_last_layers".

```py
payload = {
    "prompt": "cirno",
    "steps": 20,
    "override_settings" = {
        "sd_model_checkpoint": "Anything-V3.0-pruned",
        "CLIP_stop_at_last_layers": 2,
    }
}
```
So in this case, when I send the payload, I should get a "cirno" at 20 steps, using the "Anything-V3.0-pruned" model with the CLIP skip at 2.

For certain settings or situations, you may want your changes to stay. For that you can post to the `/sdapi/v1/options` API endpoint
We can use what we learned so far and set up the code easily for this. Here is an example:
```py
url = "http://127.0.0.1:7860"

option_payload = {
    "sd_model_checkpoint": "Anything-V3.0-pruned",
    "CLIP_stop_at_last_layers": 2
}

response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)
```
After sending this payload to the API, the model should swap to the one I set and set the CLIP skip to 2. Reiterating, this is different from `override_settings` because this change will persist, while `override_settings` is for a single request.

Note that if you're changing `sd_model_checkpoint`, the value can be the name of the checkpoint as it appears in the web UI, the filename (with or without extension), or the hash.

-----

This is as of commit [47a44c7](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/47a44c7e421b98ca07e92dbf88769b04c9e28f86)

For a more complete implementation of a frontend, my Discord bot is [here](https://github.com/Kilvoctu/aiyabot) if anyone wants to look at it as an example. Most of the action happens in stablecog.py. There are many comments explaining what each code does.

------

This guide can be found in [discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/3734) page.

Also, check out this python API client library for webui: https://github.com/mix1009/sdwebuiapi
Using custom scripts/extensions example: [here](https://github.com/mix1009/sdwebuiapi/commit/fe269dc2d4f8a98e96c63c8a7d3b5f039625bc18)