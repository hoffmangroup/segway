#!/usr/bin/env bash

#Test segway-layer command

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

TMPDIR="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"
echo >&2 "entering directory $TMPDIR"
cd "$TMPDIR"

# XXX: this now requires bedToBigBed. could refactor to look for the
# executable, but this seems odd to do in a test. - JH

segway-layer -b segway.layered.bb -m ../mnemonics < ../segway.bed > segway.layered.bed

cd ..

../compare_directory.py ../segway_layer/touchstone ../segway_layer/${TMPDIR#"./"}
