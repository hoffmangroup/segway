#!/usr/bin/env bash

## test_all.sh: run all tests

## $Revision$
## Copyright 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

set -o nounset -o pipefail -o errexit

if [[ $# != 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo usage: "$0"
    exit 2
fi

# This may be used later for specifying a subset of tests
# for dir in data simpleseg simpleconcat; do
#     "$dir/run.sh"
# done

TEST_ROOT="$(pwd)"

exit_status=0
# Avoid creating a new subshell to get an error exit status by putting the
# search commands for "run.sh" in a process substitution (file descriptor)
while read file
do
    echo "Running $(dirname $file)"
    cd "$(dirname $file)"
    # Save the exit status if any of the tests fail
    ./run.sh || { exit_status=$?; true; }
    cd $TEST_ROOT
done < <(find . -maxdepth 2 -name "run.sh" -type f | sort)

exit $exit_status
