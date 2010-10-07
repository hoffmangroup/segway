#!/usr/bin/env bash

# first argument is memory limit in kibibytes

# -c 0: no core dump files
# -v: virtual memory
# -m: per process memory limit (no effect on Linux?)
ulimit -c 0 -v "$1" -m "$1" || exit 201
shift

export TMPDIR="$(mktemp -dt segway.XXXXXXXXXX)"

on_exit ()
{
    rm -rf "$TMPDIR"
}

trap on_exit EXIT

"$@"
