Quick start
===========

Installation and configuration
==============================

1. To install Segway in your own user directories, execute this
command from :program:`bash`::

  python <(wget -O - http://noble.gs.washington.edu/proj/segway/install.py)

2. The installer will ask you some questions where to install things
and will install (if necessary) HDF5, NumPy, any other prerequisites,
and Segway.

3. Log out and back in to source the new ``~/.bashrc``.

4. If you are using SGE, your system administrator must set up a
``mem_requested`` resource for Segway to work. This can be done by
installing Segway and then running ``python -m
segway.cluster.sge_setup``.

Acquiring data
==============

5. Observation data is stored with the genomedata system.
<http://noble.gs.washington.edu/proj/genomedata/>. There is a small
Genomedata archive for testing at XXX. Retrieve it with wget::

  wget XXX

Running Segway
==============
6. train XXX::

  segway train XXX

7. identify XXX::

  segway identify XXX

8. layer XXX::

  segway-layer XXX

XXX 
     GENOMEDATADIRNAME=~hoffman/projects/encode/genomedata.all4
     EXCLUDEFILENAME=~hoffman/projects/encode/blacklist.female.bed.gz
     TRACKNAMES="DNaseI.K562.Crawford H3K36me3.K562 H3K27me3.K562"
     TRAINDIRNAME=20091117
     IDENTIFYDIRNAME=20091117.identify.20091117

     # converts every word or line to one line with -t at the beginning
     TRACKSPEC=$(echo $TRACKNAMES | perl -pe 's/(^| )/\1-t /g')

     COMMONSPEC="--num-labels=8 $TRACKSPEC \
         --seg-table=/homes/hoffman/src/projects/encode/seg_table_min_100.tab"

     ## training on 9 ENCODE regions
     REGIONSPEC="--include-coords=/homes/hoffman/projects/encode/regions.manual.1.tab
     \
         --exclude-coords=$EXCLUDEFILENAME"

     segway $COMMONSPEC $REGIONSPEC \
         -d "$TRAINDIRNAME" \
         --keep-going --prior-strength=1000 \
         --random-starts=10 \
         --no-posterior --no-identify \
         "$GENOMEDATADIRNAME"

     ## segment identification on 44 ENCODE regions using the discovered
     parameters
     REGIONSPEC="--include-coords=/homes/hoffman/projects/encode/encodeRegions.txt.gz
     \
         --exclude-coords=$EXCLUDEFILENAME"

     segway $COMMONSPEC $REGIONSPEC \
         -d "$IDENTIFYDIRNAME" \
         -i "$TRAINDIRNAME/params/input.master" \
         -p "$TRAINDIRNAME/params/params.params" \
         -s "$TRAINDIRNAME/segway.str" \
         --no-train --no-posterior \
         "$GENOMEDATADIRNAME"

   segway-layer "$IDENTIFYDIRNAME/segway.bed.gz" "$IDENTIFYDIRNAME/segway.layered.bed.gz"

Results
=======

10. If this all works you will end up with two interesting output
files. First is XXX/segway.bed.gz which has each segment
as a separate line in the BED file. The other is
$IDENTIFYDIRNAME/segway.layered.bed.gz which is designed for easier
visualization on a genome browser, by making thick lines where a
segment is defined and thin lines where it is not defined. This is not
as easy to parse, but it is more useful visually.

11. You can also perform analysis of the segmentation with segtools
<http://noble.gs.washington.edu/proj/segtools/>.
