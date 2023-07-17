#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$(which docker-compose)" ]; then
	compose='docker-compose'
else
	compose='docker compose'
fi

$compose up -d

echo -e "\nWait for the UI to start then point your browser to: http://localhost:$(docker compose ps --status running --format json | sed -e 's/.*"PublishedPort":\([0-9]\+\),.*/\1/g') \n\n"
echo "To stop showing the logs press CTRL-C"
$compose logs -f
