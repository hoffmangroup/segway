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

TMPDIR="$(mktemp -dp . test.XXXXXX)"

cd "$TMPDIR"

segway --num-labels=4 --input-master=../data/input.master train \
    ../data/test.genomedata traindir
segway identify ../data/test.genomedata traindir identifydir

# diff
diff --exclude=.svn -r -u ../data/traindir traindir > traindir.diff || true
diff --exclude=.svn -r -u ../data/identifydir identifydir > identifydir.diff \
    || true

# XXX: need to exclude differences in UUIDs/dates. one way would be to
# write a copy of these directories that replaces UUID with <UUID>,
# etc.
