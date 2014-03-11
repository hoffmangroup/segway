This is a set of simple data files that should have a
predictable segmentation.  They define a region
chr1	0	8000
and two tracks, "testtrack1" and "testtrack2".  The region should
be segmented using 4 labels, in which case it should get the
segmentation:
01230123
where each number represents a 1000-bp segment with the 
corresponding label.

testtrack1 and testtrack2 each take the values 0 or 1.

The sequence is "cgcgcg..." in the label-0 segment and 
"atatat..." elsewhere.


TODO:

Create identify recover test. Currently only have a training recovery test. Identify only generates one cluster job, so creating an identify recovery option may require a new, larger, genomedata archive.
