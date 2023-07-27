#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$(which docker-compose)" ]; then
	compose='docker-compose'
else
	compose='docker compose'
fi

cp ../webui.sh .

$compose build
