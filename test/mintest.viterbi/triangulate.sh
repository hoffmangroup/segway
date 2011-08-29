#!/usr/bin/env bash

rm *.trifile *.trifile_bak*

gmtkTriangulate \
    -cppCommandOptions "-DMISSING_FEATURE_SCALED_DIAG_GAUSSIAN" \
    -outputTriangulatedFile segway.str.missingfeaturescaleddiaggaussian.trifile \
    -strFile segway.str \
    -verbosity 0

gmtkTriangulate \
    -outputTriangulatedFile segway.str.diaggaussian.trifile \
    -strFile segway.str \
    -verbosity 0
