#!/usr/bin/env bash

## test_all.sh: run all tests

## $Revision$
## Copyright 2012 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset -o pipefail -o errexit

if [[ $# != 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo usage: "$0"
    exit 2
fi

# This may be used later for specifying a subset of tests
# for dir in data simpleseg simpleconcat; do
#     "$dir/run.sh"
# done

TEST_ROOT="$(pwd)"

find -maxdepth 2 -name "run.sh" -type f | while read file
do
    echo "Running $(dirname $file)"
    cd "$(dirname $file)"
    ./run.sh || true
    cd $TEST_ROOT
done