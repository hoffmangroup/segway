#!/usr/bin/env bash

## run.sh: test segway

## $Revision: -1 $

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

TMPDIR="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"

echo >&2 "entering directory $TMPDIR"
cd "$TMPDIR"

# XXX
MEM_PROGRESSION="0.5,1.0,2.0,2.5"
MU="0.1"
NU="0.01"
GAMMA="10"
NEIGH="5"
echo "mu=$MU; nu=$NU; gamma=$GAMMA" > mp_params

source $HOME/mprc.sh
echo "segway is: $(which segway)"

pipeline="/net/noble/vol2/home/maxwl/ssl/bin/segway_pipeline.py"
realisticmp_home="/net/noble/vol2/home/maxwl/segtest/realisticmp"

echo "starting mp pipeline..."
$pipeline \
    --data-type=none \
    --num-labels=4 \
    --resolution=1 \
    --measure-prop \
    --mu=$MU \
    --nu=$NU \
    --gamma=$GAMMA \
    --num-neighbors=$NEIGH \
    --size=synthetic \
    --seed-segs=realisticmp \
    --genomedata="$realisticmp_home/genomedata" \
    --tracks="$realisticmp_home/tracks.txt" \
    "xxx" \
    workdir-mp
   

echo "starting normal pipeline..."
$pipeline \
    --data-type=none \
    --num-labels=4 \
    --resolution=1 \
    --size=synthetic \
    --seed-segs=realisticmp \
    --genomedata="$realisticmp_home/genomedata" \
    --tracks="$realisticmp_home/tracks.txt" \
    "xxx" \
    workdir-normal
   


