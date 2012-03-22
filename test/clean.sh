#!/usr/bin/env bash

## clean.sh: remove test temporary files

## $Revision$
## Copyright 2012 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset -o pipefail -o errexit

if [[ $# != 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo usage: "$0"
    exit 2
fi

filenames="$(find . -name 'test-*' -type d)"

echo "$filenames"
echo -ne "OK to delete these files (yes/no)? "

read ok

if [ "$ok" == "yes" ]; then
    xargs rm -rv <<< "$filenames"
fi
