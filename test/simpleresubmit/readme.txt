The make-genomedata.sh script creates the simpleseg.genomedata genomedata archive from testtrack1A.bedgraph,  testtrack1B.bedgraph, chr1.fa

run.sh creates a temporary directory and runs segway against simpleresubmit.genomedata, generating 4 labels. It tests for job resubmission in cases of low memory.
