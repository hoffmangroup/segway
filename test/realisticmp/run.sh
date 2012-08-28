#!/bin/bash
set -o nounset -o pipefail -o errexit -o verbose -o xtrace


REALISTIC_MP_HOME="$HOME/segtest/realisticmp"
MP_GRAPH_EXEC=/net/noble/vol4/noble/user/maxwl/noblesvn/encode/projects/measure-prop/bin/make_mp_segway_graph.py
MAKE_TRACK_VALS_EXEC=$REALISTIC_MP_HOME/make-track-vals.py


WORKDIR=$1
mkdir -p $WORKDIR
echo >&2 "entering directory $WORKDIR"
cd "$WORKDIR"

DATADIR=data

MU=0.1
NU=0.01
GAMMA=10.0
NEIGHBORS=10
SEG_LEN=10
WINDOW_LEN=20
NUM_WINDOWS=1
VARIANCE=0.5
MP_ITERS=1
MP_AM_ITERS=10
TRAIN_ARG=both
NUM_INSTANCES=2
REUSE_EVIDENCE="False"
FREQS="0.25,0.25,0.25,0.25"

if [ ! -d $DATADIR ]; then
    mkdir -p $DATADIR

    $MAKE_TRACK_VALS_EXEC $DATADIR --variance=$VARIANCE --seg-len=$SEG_LEN \
      --window-len=$WINDOW_LEN --num-windows=$NUM_WINDOWS --freqs=$FREQS

    $MP_GRAPH_EXEC ind-sim $DATADIR/include-coords.bed 1 $DATADIR/mp_graph_${NEIGHBORS} --num-neighbors=$NEIGHBORS --segmentation=$DATADIR/correct_seg.bed.gz
fi

SEGWAY_WORKDIR="workdir"

$REALISTIC_MP_HOME/segway-wrapper.sh $SEGWAY_WORKDIR $(pwd)/$DATADIR $MU $NU $GAMMA $NEIGHBORS $MP_ITERS $MP_AM_ITERS $REUSE_EVIDENCE $NUM_INSTANCES $TRAIN_ARG




