#!/usr/bin/env bash
set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

MU="0.1"
NU="0.01"
GAMMA="1"
NUM_NEIGHBORS="50"
VARIANCE=1

WORKDIR="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"
chmod g+rx $WORKDIR

mkdir $WORKDIR/data
./make-track-vals.py $WORKDIR/data

./run.sh $WORKDIR $WORKDIR/data $MU $NU $GAMMA $NUM_NEIGHBORS $VARIANCE
