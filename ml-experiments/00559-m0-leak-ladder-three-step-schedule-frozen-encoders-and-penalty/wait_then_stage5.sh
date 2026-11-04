#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
sleep 90
while pgrep -f "/stage[234].sh" >/dev/null || pgrep -f "wait_then_stage[34]" >/dev/null; do sleep 15; done
exec $D/stage5.sh
