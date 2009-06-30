======================
 Segway documentation
======================
:Author: Michael M. Hoffman <mmh1 at washington dot edu>
:Organization: University of Washington
:Address: Department of Genome Sciences, PO Box 355065, Seattle, WA 98195-5065, United States of America
:Copyright: 2009 Michael M. Hoffman

For a conceptual overview see the paper:

Michael M. Hoffman, Jeff A. Bilmes, William Stafford Noble. Segway: a
dynamic Bayesian network for genomic segmentation. In preparation.

Michael <mmh1 at washington dot edu> can send you a copy of the latest
manuscript.

The workflow
============
Segway requires genomic data in ``genomedata`` format, and with a
single command-line will--

  1. generate an unsupervised segmentation model and initial
     parameters appropriate for this data;
  2. train parameters of the model starting with the initial parameters;
  3. identify segments in this data with the model; and
  4. calculate posterior probability for each possible segment label
     at each position.

It is also possible to run each of these steps independently. For
example, you may want to specify your own model (including models for
tasks quite unlike segmentation), train it on one dataset, and then
perform posterior decoding on another dataset, skipping the identify
stage. This is easily possible.

Generating the model
====================

Segway generates a model (``segway.str``) and initial parameters
(``input.master``) appropriate to a dataset using the GMTKL
specification language and the GMTK master parameter file format. Both
of these are described more fully in the GMTK documentation (cite),
and the default structure and starting parameters are described more
fully in the Segway article.

You can tell Segway just to generate these files and not to perform
any inference using the ``--dry-run`` option.

You can use the ``--num-labels`` option to specify the number of
possible segment labels to use in the model. Using ``--num-starts``
will generate multiple copies of the ``input.master`` file, named
``input.0.master``, ``input.1.master``, and so on, with different
randomly picked initial parameters. You may substitute your own
``input.master`` files but I recommend starting with a
Segway-generated template. This will help avoid some common pitfalls.
In particular, if you are going to perform training on your model, you
must ensure that the ``input.master`` file retains the same ``#ifdef``
structure for parameters you wish to train. Otherwise, the values
discovered after one round of trainin will not be used in subsequent
rounds, or in the identify or posterior stages.

XXX mention --num-labels is a range and will result in <num_starts>
times the length of this range different random starts.

The model may be generated using a normal distribution for continuous
observed tracks (``--distribution=normal``, the default), or a gamma
distribution (``--distribution=gamma``). The ideal methodology for
setting gamma parameter values is less well-understood, and it also
requires an unreleased version of GMTK. I recommend the use of the
default) in most cases.

XXX add arcsinh_normal similar to log

You may specify a subset of tracks using the ``--trackname`` option
which may be repeated. For example:

  XXX add example

It is very important that you always specify the same ``--trackname``
options at all stages in the Segway workflow. There is also a special
track name, ``dinucleotide``. When you specify
``--trackname=dinucleotide``, Segway will create a track containing
the dinucleotide that starts at a particular position. This can help
in modeling CpG or G+C bias.

XXX cover min-seq-len
XXX cover --prior-strength

Distributed computing
=====================
Segway can currently perform training and identification tasks only
using a cluster controllable with the DRMAA (cite) interface. I have
only tested it against Sun Grid Engine, but it should be possible to
work with other DRMAA-compatible distriuted computing systems, such as
Platform LSF, PBS, Condor, (XXXcomp add others). If you are interested
in using one of these systems, please contact me so we can correct all
the fine details.

Training
========
Most users will generate the model at training time, but to specify
your own model there are the ``--structure=<filename>`` and
``--input-master=<filename>`` options.

Training can be a time-consuming process. You may wish to train only
on a subset of your data. To facilitate this, there is an
``--include-regions=<filename>`` option which specifies a BED file
containing a list of regions to limit to. For example, the ENCODE Data
Coordination Center at University of Califronia Santa Cruz keeps the
coordinates of the ENCODE pilot regions in this format at XXXcomp. For
human whole-genome studies, these regions have nice properties since
they mark 1 percent of the genome, and were carefully picked to
include a variety of different gene densities, and a number of more
limited studies provide data just for these regions.

Memory usage
============

DBN inference can be quite memory intensive with the long sequences
found in genomic data, so Segway first tries to measure the memory use
of a single sequence. It then uses the results of this measurement to
predict memory use for future subtasks and allocate them efficiently,
if possible.

other sections of workflow XXX

XXX add section on all other options


Python interface
================
I have designed Segway such that eventually one may call different
components directly from within Python. To do so, import the following
module:

XXXcomp table here (from the setup.py)

You can then call the appropriate module through its ``main()``
function with the same arguments you would use at the command line.
For example:

  from segway import run

  GENOMEDATA_DIRNAME = "genomedata"

  run.main("--no-identify", GENOMEDATA_DIRNAME)

All other interfaces (the ones that do not use a ``main()`` function)
to Segway code are undocumented and should not be used. If you do use
them,know that the API may change at any time without notice.
