## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%\d{4}%)-(%\d{2}%)-(%\d{2}%) (%\d{2}%):(%\d{2}%):(%\d{2}%).(%\d{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "train" "--recover=../recover/traindir/" "--include-coords=../include-coords.bed" "--tracks-from=../tracks.txt" "--max-train-rounds=3" "--num-labels=4" "../simpleseg.genomedata" "traindir"
