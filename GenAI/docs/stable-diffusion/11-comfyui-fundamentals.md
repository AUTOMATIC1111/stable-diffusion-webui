# ComfyUI fundamentals

## Nodes and wires
- **Nodes** = operations
- **Sockets** = typed ports (MODEL, CLIP, LATENT, IMAGE, VAE)
- Data flows left → right

## Essential chain
CheckpointLoader → CLIP encode (+/-) → KSampler → VAEDecode → SaveImage

## Queue
Each **Queue Prompt** runs the graph. Cached nodes skip re-execution when inputs unchanged.

## Workflow JSON
Save/load from UI. Committed starters in `workflows/comfyui/`.

## Missing nodes
Red nodes = unknown type — do not auto-install; review custom node source first.
