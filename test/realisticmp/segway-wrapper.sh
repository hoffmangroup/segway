#!/bin/bash
#$ -cwd
#$ -V
#$ -l mem_requested=8G
set -o nounset -o pipefail -o errexit -o verbose -o xtrace

if [ $# != 11 ]; then
    echo usage: "$0 workdir datadir mu nu gamma neighbors mp_iters mp_am_iters num_instances train"
    exit 2
fi

source $HOME/mprc.sh
echo "segway is: $(which segway)"

OVERLAP="$HOME/ssl/bin/overlap.py"
POST_PLOT="$HOME/mp/bin/post-plot.py"
PARAMS_PLOT="$HOME/mp/bin/params_plot.py"
LL_PLOT="$HOME/mp/bin/ll-plot.R"
MP_OBJ_PLOT="$HOME/mp/bin/mp-obj-plot.R"

REALISTIC_MP_HOME="$HOME/segtest/realisticmp"
TRAINDIR_TEMPLATE="$REALISTIC_MP_HOME/traindir-template"
PARAMS="$REALISTIC_MP_HOME/true-params.params"
#STRUCTURE="$REALISTIC_MP_HOME/segway.str"

WORKDIR=$1
mkdir -p $WORKDIR
echo >&2 "entering directory $WORKDIR"
cd "$WORKDIR"

DATADIR=$2

MU=$3
NU=$4
GAMMA=$5
MP_ITERS=$6
MP_AM_ITERS=$7
REUSE_EVIDENCE=$9

NUM_INSTANCES=${10}
TRAIN_ARG=${11}

MEM_PROGRESSION="8,8"
RAND_SEED=19890506

cluster_arg="--cluster-opt=-l rhel=6,testing=TRUE"

echo "$DATADIR" > datadirname.txt

if [ $REUSE_EVIDENCE == "True" ]; then
    REUSE_EVIDENCE_LINE="--measure-prop-reuse-evidence"
else
    REUSE_EVIDENCE_LINE=""
fi
    
if [ $TRAIN_ARG == "train" -o $TRAIN_ARG == "both" ]; then
    TRAINDIR="traindir"
    SEGWAY_RAND_SEED=$RAND_SEED SEGWAY_CLUSTER_MODE="LOCAL" segway "$cluster_arg" \
        --measure-prop="$DATADIR/mp_graph_$NUM_NEIGHBORS" \
        --measure-prop-mu=$MU \
        --measure-prop-nu=$NU \
        --measure-prop-weight=$GAMMA \
        --measure-prop-num-iters=$MP_ITERS \
        --measure-prop-am-num-iters=$MP_AM_ITERS \
        ${REUSE_EVIDENCE_LINE} \
        --virtual-evidence="$DATADIR/empty.ve_labels" \
        --include-coords="$DATADIR/include-coords.bed" \
        --tracks-from="$DATADIR/tracknames.txt" \
        --ruler-scale=1 \
        --seg-table="$DATADIR/seg_table.bed" \
        --num-labels="4" \
        --mem-usage=$MEM_PROGRESSION \
        --distribution=norm \
        --graph-prior-strength=0.05 \
        --num-instances=$NUM_INSTANCES \
        train "$DATADIR/genomedata" $TRAINDIR
    PARAMS_LINE=
else
    ln -s $TRAINDIR_TEMPLATE ./traindir
    TRAINDIR=traindir/
    #PARAMS_LINE="--trainable-params=$PARAMS --structure=$STRUCTURE"
    PARAMS_LINE="--trainable-params=$PARAMS"
fi

if [ $TRAIN_ARG == "identify" -o $TRAIN_ARG == "both" ]; then
    SEGWAY_CLUSTER_MODE="LOCAL" segway "$cluster_arg" \
        --measure-prop="$DATADIR/mp_graph_$NUM_NEIGHBORS" \
        --measure-prop-mu=$MU \
        --measure-prop-nu=$NU \
        --measure-prop-weight=$GAMMA \
        --measure-prop-num-iters=$MP_ITERS \
        ${REUSE_EVIDENCE_LINE} \
        $PARAMS_LINE \
        --virtual-evidence="$DATADIR/empty.ve_labels" \
        --include-coords="$DATADIR/include-coords.bed" \
        --ruler-scale=1 \
        --seg-table="$DATADIR/seg_table.bed" \
        --mem-usage=$MEM_PROGRESSION \
        --distribution=norm \
        identify "$DATADIR/genomedata" $TRAINDIR identifydir
else
    SEGWAY_CLUSTER_MODE="LOCAL" segway "$cluster_arg" \
        --measure-prop="$DATADIR/mp_graph_$NUM_NEIGHBORS" \
        --measure-prop-mu=$MU \
        --measure-prop-nu=$NU \
        --measure-prop-weight=0 \
        --measure-prop-num-iters=1 \
        --measure-prop-am-num-iters=$MP_AM_ITERS \
        ${REUSE_EVIDENCE_LINE} \
        $PARAMS_LINE \
        --virtual-evidence="$DATADIR/empty.ve_labels" \
        --include-coords="$DATADIR/include-coords.bed" \
        --ruler-scale=1 \
        --seg-table="$DATADIR/seg_table.bed" \
        --mem-usage=$MEM_PROGRESSION \
        --distribution=norm \
        identify "$DATADIR/genomedata" $TRAINDIR identifydir
fi

$OVERLAP "$DATADIR/correct_seg.bed.gz" identifydir/segway.bed.gz 4 overlap > diagonal-frac

if [ $TRAIN_ARG == "train" -o $TRAIN_ARG == "both" ]; then
    $POST_PLOT traindir $DATADIR/correct_seg.bed.gz post-plots-train
fi

if [ $TRAIN_ARG == "identify" -o $TRAIN_ARG == "both" ]; then
    $POST_PLOT identifydir $DATADIR/correct_seg.bed.gz post-plots-identify
fi

$PARAMS_PLOT traindir $DATADIR params-plots

mkdir "ll-plots"
for i in $(seq 0 $(( $NUM_INSTANCES - 1 )) ); do
    Rscript $LL_PLOT traindir/log/likelihood.${i}.tab ll-plots/likelihood_${i}.png
done

mkdir "mp-obj-plots"
for i in $(seq 0 $(( $NUM_INSTANCES - 1 )) ); do
    Rscript $MP_OBJ_PLOT traindir/log/measure_prop_objective.${i}.tab traindir/log/likelihood.${i}.tab mp-obj-plots/mp_obj_${i}.png
done

