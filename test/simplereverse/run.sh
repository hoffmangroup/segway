#!/usr/bin/env bash

## test.sh: test segway
## run this from the parent

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

# seed from python -c "import random; print random.randrange(2**32)"
SEGWAY_RAND_SEED=4014068903 segway "$cluster_arg" \
    train --include-coords="../include-coords.bed" \
    --track=testtrack1A,testtrack1B --track=testtrack2A,testtrack2B \
    --num-labels=4 --reverse-world=1 \
    "../simpleconcat.genomedata" traindir

segway "$cluster_arg" identify --include-coords="../include-coords.bed" \
    "../simpleconcat.genomedata" traindir identifydir
