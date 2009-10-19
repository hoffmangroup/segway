#!/usr/bin/env bash

# no core dump files
ulimit -c 0

# XXX: add ulimit -v, ulimit -M

exec "$@"
