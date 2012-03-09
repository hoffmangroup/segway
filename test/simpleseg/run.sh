#!/usr/bin/env bash

## test.sh: test segway
## run this from the parent

## $Revision: -1 $

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

TMPDIR="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"

echo >&2 "entering directory $TMPDIR"
cd "$TMPDIR"

if [ "${SEGWAY_TEST_CLUSTER_OPT:-}" ]; then
    cluster_arg="--cluster-opt=$SEGWAY_TEST_CLUSTER_OPT"
else
    cluster_arg="--cluster-opt="
fi

set -x

# seed from python -c "import random; print random.randrange(2**32)"
SEGWAY_RAND_SEED=1498730685 segway "$cluster_arg" \
    --include-coords=../simpleseg/include-coords.bed \
    --tracks-from=../simpleseg/tracks.txt --num-labels=4 \
    train ../simpleseg/simpleseg.genomedata traindir

segway "$cluster_arg" --include-coords=../simpleseg/include-coords.bed \
    identify ../simpleseg/simpleseg.genomedata traindir identifydir
