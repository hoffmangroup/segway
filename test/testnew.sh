#!/usr/bin/env bash

## testnew.sh: test segway without using existing input master
## used to generate new input masters

## $Revision$
## Copyright 2011 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset
set -o pipefail
set -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

TMPDIR="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"

cd "$TMPDIR"

segway --num-labels=4 --max-train-rounds=2 \
    train ../data/test.genomedata traindir
segway identify+posterior ../data/test.genomedata traindir identifydir

# diff
../compare_directory.py ../data/traindir traindir || true # keep going
../compare_directory.py ../data/identifydir identifydir || true
