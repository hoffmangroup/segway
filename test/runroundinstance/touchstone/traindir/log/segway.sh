## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%\d{4}%)-(%\d{2}%)-(%\d{2}%) (%\d{2}%):(%\d{2}%):(%\d{2}%).(%\d{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "train-init" "--num-labels=4" "--num-instances=2" "../test.genomedata" "traindir"
## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%\d{4}%)-(%\d{2}%)-(%\d{2}%) (%\d{2}%):(%\d{2}%):(%\d{2}%).(%\d{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "train-run-round" "../test.genomedata" "traindir"
## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%\d{4}%)-(%\d{2}%)-(%\d{2}%) (%\d{2}%):(%\d{2}%):(%\d{2}%).(%\d{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "train-run-round" "../test.genomedata" "traindir"
## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%\d{4}%)-(%\d{2}%)-(%\d{2}%) (%\d{2}%):(%\d{2}%):(%\d{2}%).(%\d{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "train-finish" "../test.genomedata" "traindir"
