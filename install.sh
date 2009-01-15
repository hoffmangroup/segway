#!/usr/bin/env bash

MODULE_SW_DIR=/net/noble/vol1/software/modules-sw
PACKAGE=segway
VERSION=0.1.0a4
PREFIX="$MODULE_SW_DIR/$PACKAGE/$VERSION/$(uname -s)/${DISTRO}/${UNAME_M}"

python setup.py install --prefix="$PREFIX" \
    --install-lib="$PREFIX/lib/python2.5" --install-scripts="$PREFIX/bin" \
    --single-version-externally-managed --record=record.txt
