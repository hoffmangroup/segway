#!/usr/bin/env bash

PACKAGE=$(python -c 'import setup; print setup.name')
VERSION=$(python -c 'import setup; print setup.__version__')
PREFIX="$MOD_NOBLESW/$PACKAGE/$VERSION/$MODULES_OS/$MODULES_REL/$MODULES_MACH"

svn commit
svn cp . $(svn info | grep "^URL: " \
    | perl -pe 's/^URL: //; s/trunk$/tags\/release-/')$VERSION

python setup.py release

python setup.py install --prefix="$PREFIX" \
    --install-lib="$PREFIX/lib/python$PYTHON_VERSION" \
    --install-scripts="$PREFIX/bin" \
    --single-version-externally-managed --record=dist/record.txt

echo "increment the release"
emacs setup.py
