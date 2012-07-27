#!/usr/bin/env bash
set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

WORKDIR="$(mktemp -dp . "alltest-$(date +%Y%m%d).XXXXXX")"
chmod g+rx $WORKDIR
cd $WORKDIR

stats="stats.txt"

echo -e "mu\tnu\tgamma\tneigh\tvar\tdiagonal-frac" > $stats
chmod g+r $stats


for VARIANCE in 0.5 1 3; do 
    echo "Making data..."
    DATADIR="data-$VARIANCE"
    mkdir $DATADIR
    ( ../make-track-vals.py $DATADIR --variance=$VARIANCE > make-track-vals-out.txt ) 2> make-track-vals-err.txt
    for NU in 0.0 0.1; do
        for GAMMA in 0.0 1.0 10.0; do
            for NUM_NEIGHBORS in 5 50 500; do
                for MU in 0.1 1.0; do
                    DIR=workdir-mu${MU}-nu${NU}-gamma${GAMMA}-neigh${NUM_NEIGHBORS}-var${VARIANCE}
                    mkdir $DIR
                    chmod g+rx $DIR
                    date
                    echo "Starting: $DIR"
                    ( ../run.sh $DIR $DATADIR $MU $NU $GAMMA $NUM_NEIGHBORS $VARIANCE > $DIR/out.txt ) 2> $DIR/err.txt
                    echo -e "$MU\t$NU\t$GAMMA\t$NUM_NEIGHBORS\t$VARIANCE\t$(cat $DIR/diagonal-frac-mp)" >> $stats
                done
            done
        done
    done
done
