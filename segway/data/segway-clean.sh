#!/usr/bin/env bash

## segway-clean.sh: clean up temporary files on LSF

## $Revision$
## Copyright 2011 Michael M. Hoffman <mmh1@uw.edu>

set -o nounset
set -o pipefail
set -o errexit

rm -rf -- "$@"

if [ "${LSB_JOBID:-}" ]; then
    # have to add ".post" so mktemp doesn't complain when it already exists
    POST_TMPDIR="$(mktemp -dt "segway.$LSB_JOBID.post")"
    rm -rf "${POST_TMPDIR%%.post}"*
fi
