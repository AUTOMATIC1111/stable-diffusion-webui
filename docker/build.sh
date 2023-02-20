#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$(which dokcer-compose)" ]; then
	compose='docker-compose'
else
	compose='docker compose'
fi

$compose build
