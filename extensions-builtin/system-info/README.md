# Info Tab extensions for SD Automatic WebUI

Creates a top-level **Info** tab in Automatic WebUI with 

State & memory info are auto-updated every second if tab is visible (no updates are performed when tab is not visible)  
All other information is updated once upon WebUI load and can be force refreshed if required  

## Current information:
- Version
- Current Model & VAE
- Current State
- Current Memory statistics

## System data:
- Platform details
- Torch & CUDA details
- Active CMD flags such as `low-vram` or `med-vram`
- Versions of critical libraries
- Versions of dependent repositories

  ![screenshot](system-info.jpg)

## Models
- Models
- Hypernetworks
- Embeddings

  ![screenshot](system-info-models.jpg)

## Info Object
- System object is available as JSON for quick passing of information

  ![screenshot](system-info-json.jpg)
