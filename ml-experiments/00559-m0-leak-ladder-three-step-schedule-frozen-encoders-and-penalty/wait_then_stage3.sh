#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "/stage2.sh" >/dev/null; do sleep 15; done
exec $D/stage3.sh
