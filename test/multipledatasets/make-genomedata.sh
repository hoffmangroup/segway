# 1A and 1B are the same!

genomedata-load --sequence=chr1.fa \
    --track=testtrack1A=testtrack1A.bedgraph \
    --file-mode --verbose track1a.genomedata

genomedata-load --sequence=chr1.fa \
    --track=testtrack2A=testtrack2A.bedgraph \
    --track=testtrack3A=testtrack2A.bedgraph \
    --file-mode --verbose track23a.genomedata

genomedata-load --sequence=chr1.fa \
    --track=testtrack1B=testtrack1B.bedgraph \
    --file-mode --verbose track1b.genomedata

genomedata-load --sequence=chr1.fa \
    --track=testtrack2B=testtrack2B.bedgraph \
    --track=testtrack3B=testtrack2B.bedgraph \
    --file-mode --verbose track23b.genomedata
