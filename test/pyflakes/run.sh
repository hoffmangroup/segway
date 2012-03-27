#!/usr/bin/env bash

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
