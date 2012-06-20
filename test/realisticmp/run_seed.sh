#!/usr/bin/env bash

## run.sh: test segway

## $Revision: -1 $

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

# XXX
SEGWAY_TEST_CLUSTER_OPT="-l rhel=6,testing=TRUE"

if [ "${SEGWAY_TEST_CLUSTER_OPT:-}" ]; then
    cluster_arg="--cluster-opt=$SEGWAY_TEST_CLUSTER_OPT"
else
    cluster_arg="--cluster-opt="
fi

# XXX
MEM_PROGRESSION="0.5,1.0,1.5,2.0,2.5"

# XXX
source $HOME/mprc.sh

echo "segway is: $(which segway)"

echo "starting train...."
SEGWAY_RAND_SEED=19890806 segway "$cluster_arg" \
    --virtual-evidence="empty.ve_labels" \
    --include-coords="include-coords.bed" \
    --tracks-from="tracks.txt" \
    --ruler-scale=1 \
    --seg-table=seg_table.bed \
    --num-labels="4" \
    --mem-usage=$MEM_PROGRESSION \
    train genomedata traindir-seed

echo "starting identify...."
segway "$cluster_arg" \
    --virtual-evidence="empty.ve_labels" \
    --include-coords="include-coords.bed" \
    --ruler-scale=1 \
    --seg-table=seg_table.bed \
    --mem-usage=$MEM_PROGRESSION \
    identify genomedata traindir-seed identifydir-seed

