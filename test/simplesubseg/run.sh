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

SEGWAY_RAND_SEED=2498730688 segway "$cluster_arg" \
    train --include-coords="../include-coords.bed" \
    --num-labels=2 --num-sublabels=2\
    --tracks-from="../tracks.txt" --semisupervised="../supervision.txt" \
    "../simplesubseg.genomedata" traindir

segway "$cluster_arg" identify+posterior \
    --include-coords="../include-coords.bed" --output-label="full" \
    "../simplesubseg.genomedata" traindir identifydir-full

segway "$cluster_arg" identify+posterior \
    --include-coords="../include-coords.bed" --output-label="seg" \
    "../simplesubseg.genomedata" traindir identifydir-seg
     
cd ..

python${SEGWAY_TEST_PYTHON_VERSION:-""} ../compare_directory.py \
                        ../simplesubseg/touchstone \
                        ../simplesubseg/${testdir#"./"}
