## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%[0-9]{4}%)-(%[0-9]{2}%)-(%[0-9]{2}%) (%[0-9]{2}%):(%[0-9]{2}%):(%[0-9]{2}%).(%[0-9]{1,}%)
(%[^ ]+%)/gmtkTriangulate -cppCommandOptions (%'(?=.*-\bDCARD_SEG=4\b)(?=.*-\bDCARD_FRAMEINDEX=2000000\b)(?=.*-\bDCARD_SUBSEG=1\b)(?=.*-\bDSEGTRANSITION_WEIGHT_SCALE=1\.0\b)(?=.*-\bDCARD_SUPERVISIONLABEL=-1\b).*'%) -outputTriangulatedFile identifydir/triangulation/segway.str.4.1.trifile -strFile traindir/segway.str -verbosity 0
(%[^ ]+%)/segway-task run viterbi identifydir/viterbi/viterbi0.bed chr1 0 8000 1 0 4 1 seg ../testtrack1.bigWig,../testtrack2.bigWig asinh_norm 0,0 False None None 1 -base 3 -cVitRegexFilter '^seg$' -cliqueTableNormalize 0.0 -componentCache F -cppCommandOptions (%'(?=.*-\bDCARD_SEG=4\b)(?=.*-\bDCARD_FRAMEINDEX=2000000\b)(?=.*-\bDCARD_SUBSEG=1\b)(?=.*-\bDSEGTRANSITION_WEIGHT_SCALE=1\.0\b)(?=.*-\bDCARD_SUPERVISIONLABEL=-1\b)(?=.*-\bDINPUT_PARAMS_FILENAME=traindir/params/params\.params\b).*'%) -deterministicChildrenStore F -eVitRegexFilter '^seg$' -fmt1 binary -fmt2 binary -hashLoadFactor 0.98 -inputMasterFile traindir/params/input.master -island T -iswp1 F -iswp2 F -jtFile identifydir/log/jt_info.txt -lst 100000 -mVitValsFile - -nf1 2 -nf2 0 -ni1 0 -ni2 2 -obsNAN T -of1 identifydir/observations/float32.list -of2 identifydir/observations/int.list -pVitRegexFilter '^seg$' -strFile traindir/segway.str -triFile identifydir/triangulation/segway.str.4.1.trifile -verbosity 0 -vitCaseSensitiveRegexFilter T
