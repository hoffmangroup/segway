#!/usr/bin/env bash

## run.sh: test segway recovery option

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

#original recover data was generated with the commented-out command below
#seed from python -c "import random; print random.randrange(2**32)"
# SEGWAY_RAND_SEED=1081871764 segway "$cluster_arg" \
#     --include-coords=../include-coords.bed \
#     --tracks-from=../tracks.txt --num-labels=4 \
#     train ../simpleseg.genomedata traindir

segway "$cluster_arg" train --recover=../recover/traindir/ \
    --include-coords=../include-coords.bed --tracks-from=../tracks.txt \
    --max-train-rounds=3 --num-labels=4 ../simpleseg.genomedata traindir

segway "$cluster_arg" identify --include-coords=../include-coords.bed \
    ../simpleseg.genomedata traindir identifydir

cd ..

python${SEGWAY_TEST_PYTHON_VERSION:-""} ../compare_directory.py ../simplerecover/touchstone ../simplerecover/${testdir#"./"}
