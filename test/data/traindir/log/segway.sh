## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%[0-9]{4}%)-(%[0-9]{2}%)-(%[0-9]{2}%) (%[0-9]{2}%):(%[0-9]{2}%):(%[0-9]{2}%).(%[0-9]{1,}%)

"(%[^"]+%)/segway" "--num-labels=4" "--input-master=../data/input.master" "train" "../data/test.genomedata" "traindir"
