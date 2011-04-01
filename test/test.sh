#!/usr/bin/env bash

## test.sh: test segway

## $Revision$
## Copyright 2011 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset
set -o pipefail
set -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

segway --num-labels=3 --seg-table=data/seg_table_min_100.tab train data/GL000145.h3k27k36me3.genomedata workdir.train
segway 
