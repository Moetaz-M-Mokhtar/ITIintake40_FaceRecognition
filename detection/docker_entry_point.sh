#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    multi-model-server --start --models r50=r50.mar --model-store /root --mms-config /root/mms.config
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
