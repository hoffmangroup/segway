The make-genomedata.sh script creates the simplebadinput.genomedata genomedata archive from testtrack1A.bedgraph,  testtrack1B.bedgraph, chr1.fa

run.sh creates a temporary directory and runs segway against simplebadinput.genomedata, training only, and tries to produce an error with incorrect GMTK input.
