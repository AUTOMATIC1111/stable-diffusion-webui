
variable "RELEASE" {
    default = "v1.0.0"
}

variable "REGISTRY" {
    default = ""
}

group "default" {
  targets = ["sd-web-ui"]
}

target "sd-web-ui" {
  dockerfile = "sd.Dockerfile"
  tags       = ["${REGISTRY}${target.sd-web-ui.name}:${RELEASE}"]
  context    = "."  
  labels = {
    "org.opencontainers.image.source" = "https://github.com/webcoderz/stable-diffusion-webui"
  } 
}


#RELEASE=$(cat build/vars/BUILD_TAG) REGISTRY=$(cat build/vars/REGISTRY) docker buildx bake  --print