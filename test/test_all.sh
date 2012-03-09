#!/usr/bin/env bash

## test_all.sh: run all tests

## $Revision$
## Copyright 2012 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset -o pipefail -o errexit

if [[ $# != 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo usage: "$0"
    exit 2
fi

for dir in data simpleseg; do
    "$dir/run.sh"
done
