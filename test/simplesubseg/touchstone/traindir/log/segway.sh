## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%[0-9]{4}%)-(%[0-9]{2}%)-(%[0-9]{2}%) (%[0-9]{2}%):(%[0-9]{2}%):(%[0-9]{2}%).(%[0-9]{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "train" "--include-coords=../include-coords.bed" "--num-labels=2" "--num-sublabels=2" "--tracks-from=../tracks.txt" "--semisupervised=../supervision.txt" "../simplesubseg.genomedata" "traindir"
