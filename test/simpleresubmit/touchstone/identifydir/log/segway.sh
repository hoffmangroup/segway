## segway (%[^ ]+%) run (%[0-9a-f]{32}%) at (%[0-9]{4}%)-(%[0-9]{2}%)-(%[0-9]{2}%) (%[0-9]{2}%):(%[0-9]{2}%):(%[0-9]{2}%).(%[0-9]{1,}%)

cd "(%[^"]+%)/test-(%\d{8}%).(%[0-9a-zA-Z]{6}%)"
"(%[^"]+%)/segway" "--cluster-opt=(%[^\"]*%)" "--include-coords=(%[^"]+%)/include-coords.bed" "--mem-usage=0.346,1" "identify" "(%[^"]+%)/simpleresubmit.genomedata" "traindir" "identifydir"
