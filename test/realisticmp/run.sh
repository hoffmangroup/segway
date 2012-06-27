#!/usr/bin/env bash
set -o nounset -o pipefail -o errexit -o verbose

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

source $HOME/mprc.sh
echo "segway is: $(which segway)"

TMPDIR="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"
chmod g+rx $TMPDIR
echo >&2 "entering directory $TMPDIR"
cd "$TMPDIR"



MEM_PROGRESSION="0.5,1.0,2.0,2.5"
MU="0.1"
NU="0.01"
GAMMA="1"
NEIGH="5"
NUM_NEIGHBORS="10"
NUM_INSTANCES="1"
RAND_SEED=19890506

cluster_arg="--cluster-opt=-l rhel=6,testing=TRUE"

mkdir data
../make-track-vals.py data

echo "------------------------------------------"
echo "running normal...."
echo "------------------------------------------"

SEGWAY_RAND_SEED=$RAND_SEED segway "$cluster_arg" \
    --virtual-evidence="data/empty.ve_labels" \
    --include-coords="data/include-coords.bed" \
    --tracks-from="data/tracknames.txt" \
    --ruler-scale=1 \
    --seg-table="data/seg_table.bed" \
    --num-labels="4" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    --num-instances=$NUM_INSTANCES \
    train "data/genomedata" traindir-normal

segway "$cluster_arg" \
    --virtual-evidence="data/empty.ve_labels" \
    --include-coords="data/include-coords.bed" \
    --ruler-scale=1 \
    --seg-table="data/seg_table.bed" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    identify "data/genomedata" traindir-normal identifydir-normal

segtools-overlap "data/correct_seg.bed.gz" identifydir-normal/segway.bed.gz -o overlap-normal




echo "------------------------------------------"
echo "running seed...."
echo "------------------------------------------"
SEGWAY_RAND_SEED=$RAND_SEED segway "$cluster_arg" \
    --virtual-evidence="data/empty.ve_labels" \
    --include-coords="data/include-coords.bed" \
    --tracks-from="data/tracknames.txt" \
    --ruler-scale=1 \
    --seg-table="data/seg_table.bed" \
    --num-labels="4" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    --num-instances=$NUM_INSTANCES \
    train "data/genomedata-clean" traindir-seed

segway "$cluster_arg" \
    --virtual-evidence="data/empty.ve_labels" \
    --include-coords="data/include-coords.bed" \
    --ruler-scale=1 \
    --seg-table="data/seg_table.bed" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    identify "data/genomedata-clean" traindir-seed identifydir-seed

segtools-overlap "data/correct_seg.bed.gz" identifydir-seed/segway.bed.gz -o overlap-seed

echo "------------------------------------------"
echo "making MP files...."
echo "------------------------------------------"
/net/noble/vol4/noble/user/maxwl/noblesvn/encode/projects/measure-prop/bin/make_mp_segway_graph.py \
    ind-sim \
    "data/include-coords.bed" \
    1 \
    "data/mp_graph" \
    "--num-neighbors=$NUM_NEIGHBORS" \
    "--segmentations=identifydir-seed/segway.bed.gz"

echo "------------------------------------------"
echo "Running MP...."
echo "------------------------------------------"

SEGWAY_RAND_SEED=$RAND_SEED segway "$cluster_arg" \
    --measure-prop="data/mp_graph" \
    --measure-prop-mu=$MU \
    --measure-prop-nu=$NU \
    --measure-prop-weight=$GAMMA \
    --virtual-evidence="data/empty.ve_labels" \
    --include-coords="data/include-coords.bed" \
    --tracks-from="data/tracknames.txt" \
    --ruler-scale=1 \
    --seg-table="data/seg_table.bed" \
    --num-labels="4" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    --num-instances=$NUM_INSTANCES \
    train "data/genomedata" traindir-mp

segway "$cluster_arg" \
    --measure-prop="data/mp_graph" \
    --measure-prop-mu=$MU \
    --measure-prop-nu=$NU \
    --measure-prop-weight=$GAMMA \
    --virtual-evidence="data/empty.ve_labels" \
    --include-coords="data/include-coords.bed" \
    --ruler-scale=1 \
    --seg-table="data/seg_table.bed" \
    --mem-usage=$MEM_PROGRESSION \
    --distribution=norm \
    identify "data/genomedata" traindir-mp identifydir-mp

segtools-overlap "data/correct_seg.bed.gz" identifydir-mp/segway.bed.gz -o overlap-mp




