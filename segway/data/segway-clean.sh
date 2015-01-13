#!/usr/bin/env bash

## segway-clean.sh: clean up temporary files on LSF

## $Revision$
## Copyright 2011, 2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

set -o nounset -o pipefail -o errexit

if [ "$#" -gt 0 ]; then
    rm -rf -- "$@"
fi

if [ "${LSB_JOBID:-}" ]; then
    # get location used by mktemp
    POST_TMPDIR="$(mktemp -dt "segway.$LSB_JOBID.XXXXXXXXXX")"
    rm -rf "${POST_TMPDIR%.*}"*
fi
