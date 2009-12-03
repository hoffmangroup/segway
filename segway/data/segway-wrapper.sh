#!/usr/bin/env bash

# first argument is memory limit in kibibytes

# -c 0: no core dump files
ulimit -c 0 -v "$1" -M "$1" || exit 201
shift

export TMPDIR="$(mktemp -dt segway.XXXXXXXXXX)"

die ()
{
    rm -rf "$TMPDIR"
    exit 200
}

trap die 15 10 12 2 1

"$@"
RETVAL="$?"

rm -rf "$TMPDIR"

exit "$RETVAL"
