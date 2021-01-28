=================
 Quickstart Guide
=================

Installation and configuration
==============================

With Bioconda
-------------
    conda install segway

Generic installation
----------------

1. To install Segway first install `GMTK
   <https://github.com/melodi-lab/gmtk/releases>`_,
   and install `HDF5 <http://www.hdfgroup.org/downloads/index.html>`_,
   then run this command from :program:`bash`::

     pip install segway

2. If you are using SGE, your system administrator must set up a
   ``mem_requested`` resource for Segway to work. This can be done by
   installing Segway and then running ``python -m
   segway.cluster.sge_setup``.

Acquiring data
==============

3. Observation data is stored with the genomedata system.
   <http://pmgenomics.ca/hoffmanlab/proj/genomedata/>. There is a small
   Genomedata archive for testing that comes with Segway, that is used
   in the below steps. You can get it using::

     wget http://pmgenomics.ca/hoffmanlab/proj/segway/2011/test.genomedata

Running Segway
==============
4. Use the ``segway train`` command to discover patterns in the test
   data. Here, we specify that we want Segway to discover four unique
   patterns::

     segway train --num-labels=4 test.genomedata traindir

5. Use the ``segway identify`` command to create the segmentation,
   which partitions the genome into regions labeled with one of the
   four discovered patterns::

     segway identify test.genomedata traindir identifydir

.. note::

  This example spawns jobs that will run sequentially due to small
  segment size. See the :option:`--split-sequences` option for
  dividing segments into smaller pieces.

Results
=======

6. The ``identifydir/segway.bed.gz`` file has each segment as a
   separate line in the BED file, and can be used for further
   processing.

7. The ``identifydir/segway.layered.bed.gz`` file is designed for
   easier visualization on a genome browser. It has thick lines where
   a segment is present and thin lines where it is not. This is not as
   easy for a computer to parse, but it is more useful visually.

8. You can also perform further analysis of the segmentation and
   trained parameters using Segtools <http://pmgenomics.ca/hoffmanlab/proj/segtools/>.
