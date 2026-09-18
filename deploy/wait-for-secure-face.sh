#!/bin/sh
set -eu

attempt=0
while [ "$attempt" -lt 60 ]; do
    if /usr/bin/curl --fail --silent --show-error --max-time 2 \
        http://127.0.0.1:8000/ >/dev/null; then
        echo "[wait] SecureFace-RX is ready on 127.0.0.1:8000"
        exit 0
    fi
    attempt=$((attempt + 1))
    sleep 1
done

echo "[wait] SecureFace-RX did not become ready within 60 seconds" >&2
exit 1
