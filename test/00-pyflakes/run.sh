#!/usr/bin/env bash

# Checks all .py files in segway install using ./flymake-pyflakes script
# which itself should run pyflakes on all of the python files.
# $Revision$
# Copyright 2010, 2011, 2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

set -o nounset -o pipefail -o errexit

if [ $# != 0 ]; then
    echo usage: "$0"
    exit 2
fi

if ! python${SEGWAY_TEST_PYTHON_VERSION:-""} -c 'import pyflakes'; then
    echo "ERROR: Cannot find pyflakes installation."
    echo "skipping pyflakes check"
    exit 127
fi

echo "Checking with Pyflakes"

find ../.. -wholename ../../dist -prune -or -wholename ../../build -prune \
    -or -wholename '../../ez_setup.py' -prune \
    -or -name "*.py" -exec ./flymake-pyflakes.sh {} +
