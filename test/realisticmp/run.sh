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
mkdir -p $DATADIR

MU=0.1
NU=0.0
GAMMA=100.0
NEIGHBORS=10
SEG_LEN=10
WINDOW_LEN=100
NUM_WINDOWS=1
VARIANCE=0.5
MP_ITERS=1
TRAIN_ARG=both
FREQS="0.25,0.25,0.25,0.25"

$MAKE_TRACK_VALS_EXEC $DATADIR --variance=$VARIANCE --seg-len=$SEG_LEN \
  --window-len=$WINDOW_LEN --num-windows=$NUM_WINDOWS --freqs=$FREQS

$MP_GRAPH_EXEC ind-sim $DATADIR/include-coords.bed 1 $DATADIR/mp_graph_${NEIGHBORS} --num-neighbors=$NEIGHBORS --segmentation=$DATADIR/correct_seg.bed.gz

SEGWAY_WORKDIR="workdir"

$REALISTIC_MP_HOME/segway-wrapper.sh $SEGWAY_WORKDIR $(pwd)/$DATADIR $MU $NU $GAMMA $NEIGHBORS $MP_ITERS $TRAIN_ARG




