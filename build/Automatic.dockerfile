FROM alpine:3.17

RUN apk update && apk add --no-cache python3 python3-dev

WORKDIR home/stable-diffusion-webui

COPY configs embeddings extensions extensions-builtin javascript localizations models modules scripts test \
    textual_inversion_templates artists.csv *.yaml *.py webui.sh webui-user.sh ./

CMD webui.sh --api
