# 1A and 1B are the same!

genomedata-load --sequence=chr1.fa \
    --track=testtrack1A=testtrack1.bedgraph --track=testtrack2A=testtrack2.bedgraph \
    --track=testtrack1B=testtrack1.bedgraph --track=testtrack2B=testtrack2.bedgraph \
    --file-mode --verbose simpleconcat.genomedata
