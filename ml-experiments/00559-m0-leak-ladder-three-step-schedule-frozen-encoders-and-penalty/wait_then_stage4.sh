#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
sleep 60
while pgrep -f "/stage[23].sh" >/dev/null || pgrep -f "wait_then_stage3" >/dev/null; do sleep 15; done
exec $D/stage4.sh
