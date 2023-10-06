#!/usr/bin/env bash

# Copyright 2010, 2011, 2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

if ! python -c 'import flake8'; then
    echo "ERROR: Cannot find flake8 installation."
    echo "skipping flake8 check"
    exit 127
fi

echo "Checking with flake8"

cd ../..
flake8
