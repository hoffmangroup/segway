#!/bin/bash

if [ -e genomedata ]; then
    echo "use clean.sh first"
    exit
fi

./make-track-vals.py

./run_seed.sh

segtools-overlap correct_seg.bed.gz identifydir-seed/segway.bed.gz -o identifydir-seed/overlap

./run.sh


