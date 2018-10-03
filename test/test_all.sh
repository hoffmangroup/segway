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

# Clear existing archives created by previous failures
rm */*-changes.tar.gz || true

# Use a non-default system python if set
PYTHON_PROG=python${SEGWAY_TEST_PYTHON_VERSION:-""}

# Run unit tests
$PYTHON_PROG unit_tests.py

exit_status=0
# Avoid creating a new subshell to get an error exit status by putting the
# search commands for "run.sh" in a process substitution (file descriptor)
# This ignores tests which do not have a run.sh such as simpleresubmit
while read file
do
    echo "Running $(dirname $file)"
    cd "$(dirname $file)"
    # Save the exit status if any of the tests fail
    ./run.sh || { exit_status=$?; true; }
    cd $TEST_ROOT
done < <(find . -maxdepth 2 -name "run.sh" -type f | sort)

#mkdir all-test-changes || true
#cp */*-changes.tar.gz all-test-changes || true
#tar -cf all-test-changes.tar all-test-changes || true
#rm -rf all-test-changes || true

exit $exit_status
