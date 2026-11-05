#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
sleep 120
while pgrep -f "/stage[2345].sh" >/dev/null || pgrep -f "wait_then_stage[345]" >/dev/null; do sleep 15; done
exec $D/stage6.sh
