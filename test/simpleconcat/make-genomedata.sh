# 1A and 1B are the same!

genomedata-load --sequence=chr1.fa \
    --track=testtrack1A=testtrack1A.bedgraph --track=testtrack2A=testtrack2A.bedgraph \
    --track=testtrack1B=testtrack1B.bedgraph --track=testtrack2B=testtrack2B.bedgraph \
    --file-mode --verbose simpleconcat.genomedata
