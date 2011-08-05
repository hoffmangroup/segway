#!/usr/bin/env bash

set -x

COMMONOPTIONS=(-fmt1 binary
    -iswp1 F
    -nf1 2
    -ni1 0

    -island T
    -lst 100000

    -pVitRegexFilter ^seg$

    -inputMasterFile input.master
    -strFile segway.str
    -verbosity 30)

# gmtkViterbi \
#     -base 3 \
#     -cppCommandOptions "-DCARD_SEG=4 -DINPUT_PARAMS_FILENAME=traindir/params/params.params -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" \
#     -fmt1 binary \
#     -fmt2 binary \
#     -inputMasterFile ../data/input.master \
#     -island T \
#     -iswp1 F \
#     -iswp2 F \
#     -jtFile identifydir/log/jt_info.txt \
#     -lst 100000 \
#     -nf1 0 \
#     -nf2 2 \
#     -ni1 2 \
#     -ni2 0 \
#     -of1 int.list \
#     -of2 float32.nonan.list \
#     -pVitRegexFilter ^seg$ \
#     -pVitValsFile viterbi.renewed.txt \
#     -strFile segway.str \
#     -triFile segway.str.diaggaussian.trifile \
#     -verbosity 30

gmtkViterbi \
    "${COMMONOPTIONS[@]}" \
    -cppCommandOptions "-DCARD_SEG=4 -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" \
    -obsNAN F \
    -of1 float32.nonan.list \
    -fmt2 binary \
    -iswp2 F \
    -nf2 0 \
    -ni2 2 \
    -of2 int.list \
    -triFile segway.str.diaggaussian.trifile \
    -pVitValsFile viterbi.diaggaussian.nonan.txt

gmtkViterbi \
    "${COMMONOPTIONS[@]}" \
    -cppCommandOptions "-DCARD_SEG=4 -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" \
    -obsNAN T \
    -of1 float32.list \
    -fmt2 binary \
    -iswp2 F \
    -nf2 0 \
    -ni2 2 \
    -of2 int.list \
    -triFile segway.str.diaggaussian.trifile \
    -pVitValsFile viterbi.diaggaussian.obsnan.txt

gmtkViterbi \
    "${COMMONOPTIONS[@]}" \
    -cppCommandOptions "-DMISSING_FEATURE_SCALED_DIAG_GAUSSIAN -DCARD_SEG=4 -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" \
    -obsNAN T \
    -of1 float32.list \
    -triFile segway.str.missingfeaturescaleddiaggaussian.trifile \
    -pVitValsFile viterbi.missingfeaturescaleddiaggaussian.txt

diff -u viterbi.old.txt viterbi.diaggaussian.nonan.txt
# diff -u viterbi.old.txt viterbi.renewed.txt
# diff -u viterbi.diaggaussian.nonan.txt viterbi.diaggaussian.obsnan.txt
diff -u viterbi.diaggaussian.obsnan.txt viterbi.missingfeaturescaleddiaggaussian.txt
