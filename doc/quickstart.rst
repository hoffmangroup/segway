Quick start
===========

Installation and configuration
==============================

1. To install Segway in your own user directories, execute this
command from :program:`bash`::

  python <(wget -O - http://noble.gs.washington.edu/proj/segway/install.py)

2. The installer will ask you some questions where to install things
and will install (if necessary) HDF5, NumPy, any other prerequisites,
and Segway. It will also tell you about changes it wants to make to
your ``~/.bashrc`` to set up your environment properly.

3. Log out and back in to source the new ``~/.bashrc``.

4. If you are using SGE, your system administrator must set up a
``mem_requested`` resource for Segway to work. This can be done by
installing Segway and then running ``python -m
segway.cluster.sge_setup``.

Acquiring data
==============

5. Observation data is stored with the genomedata system.
<http://noble.gs.washington.edu/proj/genomedata/>. There is a small
Genomedata archive for testing that comes with Segway, that is used in
the below steps.

Running Segway
==============
6. Use the ``segway train`` command to discover patterns in the test
data. Here, we specify that we want Segway to discover four unique
patterns::

  segway --num-labels=4 train test/data/test.genomedata traindir

7. Use the ``segway identify`` command to create the segmentation,
which partitions the genome into regions labeled with one of the four
discovered patterns::

  segway identify test/data/test.genomedata traindir identifydir

Results
=======

8. The ``identifydir/segway.bed.gz`` file has each segment as a
separate line in the BED file, and can be used for further processing.

9. The ``identifydir/segway.layered.bed.gz`` file is designed for
easier visualization on a genome browser. It has thick lines where a
segment is present and thin lines where it is not. This is not as easy
for a computer to parse, but it is more useful visually.

10. You can also perform further analysis of the segmentation and
trained parameters using Segtools
<http://noble.gs.washington.edu/proj/segtools/>.
