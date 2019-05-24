## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%[0-9]{4}%)-(%[0-9]{2}%)-(%[0-9]{2}%) (%[0-9]{2}%):(%[0-9]{2}%):(%[0-9]{2}%).(%[0-9]{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "--include-coords=../include-coords.bed" "--track=testtrack1A,testtrack1B" "--track=testtrack2A,testtrack2B" "--track=testtrack3A,testtrack3B" "--num-labels=4" "train" "../track1a.genomedata" "../track1b.genomedata" "../track23a.genomedata" "../track23b.genomedata" "traindir"
