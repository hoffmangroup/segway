#!/usr/bin/env bash

# first argument is memory limit in kibibytes

# -c 0: no core dump files
# -v: virtual memory
# -m: per process memory limit (no effect on Linux?)
ulimit -c 0 -v "$1" -m "$1" || exit 201

# the original temporary dir used by the submitting program (usually is /tmp)
SUBMIT_TMPDIR="$2"
shift 2

export TMPDIR="$(mktemp -dt segway.XXXXXXXXXX)"

on_exit ()
{
    rm -rf "$TMPDIR"

    # delete any arguments that begin with $SUBMIT_TMPDIR
    for ARG in "$@"; do
        if [[ "$ARG" == "$SUBMIT_TMPDIR"* ]]; then
            rm -rf "$TMPDIR" 2>/dev/null || true
        fi
    done
}

trap on_exit EXIT

"$@"
