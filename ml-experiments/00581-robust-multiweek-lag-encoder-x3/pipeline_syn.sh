#!/bin/bash
# X3-syn (X3-med fine-tuned with lag-only synthetic events on 5% of the time) and its v3+D5 control. Resumable.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for k in x3syn v3syn; do
  $D/run.sh $k fit
  $D/run.sh $k assemble
  $D/run.sh $k inject
done
echo SYN DONE
