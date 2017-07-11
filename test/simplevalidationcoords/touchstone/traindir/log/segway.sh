## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%\d{4}%)-(%\d{2}%)-(%\d{2}%) (%\d{2}%):(%\d{2}%):(%\d{2}%).(%\d{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--num-labels=4" "--max-train-rounds=2" "--include-coords=../include-coords.bed" "--minibatch-fraction=0.1" "--split-sequences=25000" "--validation-coords=../validation-coords.bed" "--cluster-opt=" "train" "../test.genomedata" "traindir"