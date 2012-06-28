#!/usr/bin/env bash
set -o nounset -o pipefail -o errexit

if [ $# != 7 ]; then
    echo usage: "$0 workdir datadir mu nu gamma neighbors variance"
    exit 2
fi

source $HOME/mprc.sh
echo "segway is: $(which segway)"

OVERLAP="$HOME/ssl/bin/overlap.py"

WORKDIR=$1
echo >&2 "entering directory $WORKDIR"
cd "$WORKDIR"

DATADIR=$2
DATADIR="../${DATADIR}"

MU=$3
NU=$4
GAMMA=$5
NUM_NEIGHBORS=$6
VARIANCE=$7

MEM_PROGRESSION="0.5,1.0,2.0,2.5"
NUM_INSTANCES="5" # XXX
RAND_SEED=19890506

cluster_arg="--cluster-opt=-l rhel=6,testing=TRUE"


#echo "------------------------------------------"
#echo "running normal...."
#echo "------------------------------------------"

#SEGWAY_RAND_SEED=$RAND_SEED segway "$cluster_arg" \
    #--virtual-evidence="$DATADIR/empty.ve_labels" \
    #--include-coords="$DATADIR/include-coords.bed" \
    #--tracks-from="$DATADIR/tracknames.txt" \
    #--ruler-scale=1 \
    #--seg-table="$DATADIR/seg_table.bed" \
    #--num-labels="4" \
    #--mem-usage=$MEM_PROGRESSION \
    #--distribution=norm \
    #--num-instances=$NUM_INSTANCES \
    #train "$DATADIR/genomedata" traindir-normal

#segway "$cluster_arg" \
    #--virtual-evidence="$DATADIR/empty.ve_labels" \
    #--include-coords="$DATADIR/include-coords.bed" \
    #--ruler-scale=1 \
    #--seg-table="$DATADIR/seg_table.bed" \
    #--mem-usage=$MEM_PROGRESSION \
    #--distribution=norm \
    #identify "$DATADIR/genomedata" traindir-normal identifydir-normal

#$OVERLAP "$DATADIR/correct_seg.bed.gz" identifydir-normal/segway.bed.gz 4 overlap-normal > diagonal-frac-normal


#echo "------------------------------------------"
#echo "running seed...."
#echo "------------------------------------------"
#SEGWAY_RAND_SEED=$RAND_SEED segway "$cluster_arg" \
    #--virtual-evidence="$DATADIR/empty.ve_labels" \
    #--include-coords="$DATADIR/include-coords.bed" \
    #--tracks-from="$DATADIR/tracknames.txt" \
    #--ruler-scale=1 \
    #--seg-table="$DATADIR/seg_table.bed" \
    #--num-labels="4" \
    #--mem-usage=$MEM_PROGRESSION \
    #--distribution=norm \
    #--num-instances=$NUM_INSTANCES \
    #train "$DATADIR/genomedata-clean" traindir-seed

#segway "$cluster_arg" \
    #--virtual-evidence="$DATADIR/empty.ve_labels" \
    #--include-coords="$DATADIR/include-coords.bed" \
    #--ruler-scale=1 \
    #--seg-table="$DATADIR/seg_table.bed" \
    #--mem-usage=$MEM_PROGRESSION \
    #--distribution=norm \
    #identify "$DATADIR/genomedata-clean" traindir-seed identifydir-seed

#$OVERLAP "$DATADIR/correct_seg.bed.gz" identifydir-seed/segway.bed.gz 4 overlap-seed > diagonal-frac-seed

echo "------------------------------------------"
echo "making MP files...."
echo "------------------------------------------"
/net/noble/vol4/noble/user/maxwl/noblesvn/encode/projects/measure-prop/bin/make_mp_segway_graph.py \
    ind-sim \
    "$DATADIR/include-coords.bed" \
    1 \
    "$DATADIR/mp_graph" \
    "--num-neighbors=$NUM_NEIGHBORS" \
    "--segmentations=$DATADIR/correct_seg.bed.gz"

echo "------------------------------------------"
echo "Running MP...."
echo "------------------------------------------"

SEGWAY_RAND_SEED=$RAND_SEED segway "$cluster_arg" \
    --measure-prop="$DATADIR/mp_graph" \
    --measure-prop-mu=$MU \
    --measure-prop-nu=$NU \
    --measure-prop-weight=$GAMMA \
    --virtual-evidence="$DATADIR/empty.ve_labels" \
    --include-coords="$DATADIR/include-coords.bed" \
    --tracks-from="$DATADIR/tracknames.txt" \
    --ruler-scale=1 \
    --seg-table="$DATADIR/seg_table.bed" \
    --num-labels="4" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    --num-instances=$NUM_INSTANCES \
    train "$DATADIR/genomedata" traindir-mp

segway "$cluster_arg" \
    --measure-prop="$DATADIR/mp_graph" \
    --measure-prop-mu=$MU \
    --measure-prop-nu=$NU \
    --measure-prop-weight=$GAMMA \
    --virtual-evidence="$DATADIR/empty.ve_labels" \
    --include-coords="$DATADIR/include-coords.bed" \
    --ruler-scale=1 \
    --seg-table="$DATADIR/seg_table.bed" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    identify "$DATADIR/genomedata" traindir-mp identifydir-mp

$OVERLAP "$DATADIR/correct_seg.bed.gz" identifydir-mp/segway.bed.gz 4 overlap-mp > diagonal-frac-mp




