#!/bin/bash
# Run the remaining phase-2 GPU jobs strictly one after another (never two GPU jobs at once).
set -e
cd "$(dirname "$0")"
PY=/root/miniforge3/envs/pp2-pseudo/bin/python
$PY repr_search.py pooled --workers 6 >> logs/repr_pooled.log 2>&1
$PY repr_search.py tf --workers 2 --seeds 0 > logs/repr_tf.log 2>&1
TOP=$($PY -c "
from common import read_results
r=sorted([x for x in read_results() if (x.get('name') or '').startswith('tf_') and x['seed']==0], key=lambda x:-x['auroc_mean'])
print(','.join(x['name'] for x in r[:2]))")
$PY repr_search.py tf --workers 2 --seeds 1,2 --only "$TOP" >> logs/repr_tf.log 2>&1
echo QUEUE_DONE >> logs/repr_tf.log
