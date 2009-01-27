#!/usr/bin/env bash

# XXX: do for each platform
PACKAGE="$(python setup.py --name)"
VERSION="$(python setup.py --version)"
PREFIX="$MOD_NOBLESW/$PACKAGE/$VERSION/$MODULES_OS/$MODULES_REL/$MODULES_MACH"
python setup.py install --prefix="$PREFIX" \
    --install-lib="$PREFIX/lib/python$PYTHON_VERSION" \
    --install-scripts="$PREFIX/bin" \
    --single-version-externally-managed --record=dist/record.txt

