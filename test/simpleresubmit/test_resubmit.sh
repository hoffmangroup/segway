#!/usr/bin/env bash

## run.sh: test segway

## $Revision: -1 $

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

testdir="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"

echo >&2 "entering directory $testdir"
cd "$testdir"

if [ "${SEGWAY_TEST_CLUSTER_OPT:-}" ]; then
    cluster_arg="--cluster-opt=$SEGWAY_TEST_CLUSTER_OPT"
else
    cluster_arg="--cluster-opt="
fi

set -x

# Set mem-usage to 30000KB in fraction of a GB
SEGWAY_RAND_SEED=1498730685 segway "$cluster_arg" \
    --include-coords="../include-coords.bed" \
    --tracks-from="../tracks.txt" --num-labels=4 \
    --mem-usage="0.030,0.031,1" \
    train "../simpleresubmit.genomedata" traindir

segway "$cluster_arg" --include-coords="../include-coords.bed" \
    --mem-usage="0.346,1" \
    identify "../simpleresubmit.genomedata" traindir identifydir

cd ..

../compare_directory.py ../simpleresubmit/touchstone ../simpleresubmit/${testdir#"./"}
