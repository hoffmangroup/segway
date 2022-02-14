#!/usr/bin/env bash

# flymake-pyflakes: run PyFlakes; set warnings appropriately
# $Revision$
# Copyright 2010, 2011 Michael M. Hoffman <michael.hoffman@utoronto.ca>

# usage: flymake-pyflakes

set -o nounset -o pipefail -o errexit

# this formulation ensures that you use your current python instead of
# whatever was installed when pyflakes was installed
python${SEGWAY_TEST_PYTHON_VERSION:-""} -c "from pyflakes.scripts.pyflakes import main; main()" "$@" \
    | perl -pe 's/([^:]+:[^:]+):(.* (?:imported but unused|is assigned to but never used))$/\1:warning:\2/'
