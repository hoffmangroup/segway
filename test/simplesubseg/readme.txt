The make-genomedata.sh script creates the simplesubseg.genomedata genomedata archive from testtrack1.bedgraph,  testtrack2.bedgraph, chr1.fa

run.sh creates a temporary directory and runs segway against simplesubseg.genomedata, generating 2 labels each with 2 sublabels. The train task is run in semi-supervised mode to ensure compatibility with semi-supervised training and the identify task.
