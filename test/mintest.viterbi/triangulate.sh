#!/usr/bin/env bash

rm *.trifile *.trifile_bak*

gmtkTriangulate \
    -cppCommandOptions "-DMISSING_FEATURE_SCALED_DIAG_GAUSSIAN -DCARD_SEG=4 -DINPUT_PARAMS_FILENAME=params.params -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" \
    -outputTriangulatedFile segway.str.missingfeaturescaleddiaggaussian.trifile \
    -strFile segway.str \
    -verbosity 0

gmtkTriangulate \
    -cppCommandOptions "-DCARD_SEG=4 -DINPUT_PARAMS_FILENAME=params.params -DCARD_FRAMEINDEX=2000000 -DSEGTRANSITION_WEIGHT_SCALE=1.0 -DCARD_SUBSEG=1" \
    -outputTriangulatedFile segway.str.diaggaussian.trifile \
    -strFile segway.str \
    -verbosity 0
