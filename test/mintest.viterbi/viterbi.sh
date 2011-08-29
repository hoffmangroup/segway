#!/usr/bin/env bash

set -x

COMMONOPTIONS=(-fmt1 binary
    -iswp1 F
    -nf1 2
    -ni1 0
    -of1 float32.list

    -fmt2 binary
    -iswp2 F
    -nf2 0
    -ni2 2
    -of2 int.list

    -obsNAN T

    -island T
    -lst 100000

    -pVitRegexFilter ^seg$

    -inputMasterFile input.master
    -strFile segway.str
    -verbosity 30)

gmtkViterbi \
    "${COMMONOPTIONS[@]}" \
    -triFile segway.str.diaggaussian.trifile \
    -pVitValsFile viterbi.diaggaussian.obsnan.txt

gmtkViterbi \
    "${COMMONOPTIONS[@]}" \
    -cppCommandOptions "-DMISSING_FEATURE_SCALED_DIAG_GAUSSIAN" \
    -triFile segway.str.missingfeaturescaleddiaggaussian.trifile \
    -pVitValsFile viterbi.missingfeaturescaleddiaggaussian.txt

diff -u viterbi.old.txt viterbi.diaggaussian.obsnan.txt
diff -u viterbi.diaggaussian.obsnan.txt viterbi.missingfeaturescaleddiaggaussian.txt
