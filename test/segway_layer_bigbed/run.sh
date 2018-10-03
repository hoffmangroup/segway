#!/usr/bin/env bash

#Test bigBed creation in segway-layer

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

testdir="$(mktemp -dp . "test-$(date +%Y%m%d).XXXXXX")"
echo >&2 "entering directory $testdir"
cd "$testdir"

if type -P bedToBigBed >/dev/null; then

    segway-layer -b segway.layered.bb -m ../mnemonics \
        < ../segway.bed > segway.layered.bed
    cd ..

    python${SEGWAY_TEST_PYTHON_VERSION:-""} ../compare_directory.py ../segway_layer_bigbed/touchstone \
        ../segway_layer_bigbed/${testdir#"./"}

else
    echo bedToBigBed not found in PATH. test skipped. >&2
fi


