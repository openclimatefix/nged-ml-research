#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "stage1.sh" >/dev/null; do sleep 15; done
exec $D/stage2.sh
