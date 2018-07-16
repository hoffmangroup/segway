#!/usr/bin/env bash

## memory limit in kibibytes
mem_limit_kb="$1"

## the original temporary dir used by the submitting program (usually is /tmp)
submit_tmpdir="$2"
stdout_filename="$3"
shift 3

# -c 0: no core dump files
# -v: virtual memory
# -m: per process memory limit (no effect on newer Linuxes)
ulimit -c 0 -v "$mem_limit_kb" -m "$mem_limit_kb" || exit 201

if [ "${LSB_JOBID:-}" ]; then
    # this way, the post-exec script can delete it easily
    export TMPDIR="$(mktemp -dt "segway.$LSB_JOBID.XXXXXXXXXX")"
else
    export TMPDIR="$(mktemp -dt segway.XXXXXXXXXX)"
fi

on_exit ()
{
    rm -rf "$TMPDIR"
}

trap on_exit EXIT

## start a subshell so that going over ulimit only kills the subshell
bash "$@" > "$stdout_filename"
