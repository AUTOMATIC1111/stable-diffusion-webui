To install custom scripts, drop them into `scripts` directory and restart the web ui.

## Advanced prompt matrix
https://github.com/GRMrGecko/stable-diffusion-webui-automatic/blob/advanced_matrix/scripts/advanced_prompt_matrix.py

It allows a matrix prompt as follows:
`<cyber|cyborg|> cat <photo|image|artistic photo|oil painting> in a <car|boat|cyber city>`

Does not actually draw a matrix, just produces pictures.

## Wildcards
https://github.com/jtkelm2/stable-diffusion-webui-1/blob/master/scripts/wildcards.py

Added script support so that prompts can contain wildcard terms (indicated by surrounding double underscores), with values instantiated randomly from the corresponding .txt file in the folder `/scripts/wildcards/`. For example:

`a woman at a cafe by __artist__ and __artist__`

will draw two random artists from `artist.txt`. This works independently on each prompt, so that one can e.g. generate a batch of 100 images with the same prompt input using wildcards, and each output will have unique artists (or styles, or genres, or anything that the user creates their own .txt file for) attached to it.
