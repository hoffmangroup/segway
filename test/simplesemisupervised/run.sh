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

SEGWAY_RAND_SEED=7908 segway "$cluster_arg" \
    --semisupervised="../semi-label.bed" \
    --resolution=10 \
    --include-coords="../include-coords.bed" \
    --tracks-from="../tracks.txt" --num-labels=4 \
    train "../simplesemi.genomedata" traindir


segway "$cluster_arg" --include-coords="../include-coords.bed" \
    identify "../simplesemi.genomedata" traindir identifydir

cd ..

../compare_directory.py ../simplesemisupervised/touchstone ../simplesemisupervised/${testdir#"./"}
