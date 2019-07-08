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

SEGWAY_RAND_SEED=1498730685 segway "$cluster_arg" \
    train --include-coords="../include-coords.bed" \
    --tracks-from="../tracks.txt" --num-labels=4 \
    "../simpleseg.genomedata" traindir 

segway "$cluster_arg" annotate-init  --include-coords="../include-coords.bed" \
    "../simpleseg.genomedata" traindir identifydir
    
    
python${SEGWAY_TEST_PYTHON_VERSION:-""} ../../compare_directory.py ../touchstone_init ./

segway "$cluster_arg" annotate-run "../simpleseg.genomedata" traindir identifydir

python${SEGWAY_TEST_PYTHON_VERSION:-""} ../../compare_directory.py ../touchstone_run ./
    
segway "$cluster_arg" annotate-finish "../simpleseg.genomedata" traindir identifydir

python${SEGWAY_TEST_PYTHON_VERSION:-""} ../../compare_directory.py ../touchstone_finish ./
