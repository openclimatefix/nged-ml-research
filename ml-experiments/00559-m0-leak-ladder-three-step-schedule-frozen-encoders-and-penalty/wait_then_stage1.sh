#!/bin/bash
D="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "python.*e00559_fit.py" >/dev/null; do sleep 10; done
exec $D/stage1.sh
