#!/usr/bin/env bash

# Checks all .py files in segway install using ./flymake-pyflakes script
# which itself should run pyflakes on all of the python files.
# $Revision$
# Copyright 2010, 2011 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

PYFLAKES_PATH=$(type -P pyflakes) || true

if [ -z $PYFLAKES_PATH ]; then
    echo "ERROR: Cannot find pyflakes installation."
    echo "skipping pyflakes check"
    exit 127
fi

find ../.. -name "*.py" -exec ./flymake-pyflakes.sh {} +
