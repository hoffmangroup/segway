======================
 Segway documentation
======================
:Author: Michael M. Hoffman <mmh1 at washington dot edu>
:Organization: University of Washington
:Address: Department of Genome Sciences, PO Box 355065, Seattle, WA 98195-5065, United States of America
:Copyright: 2009 Michael M. Hoffman

For a conceptual overview see the paper:

  Michael M. Hoffman, Jeff A. Bilmes, William Stafford Noble. Segway:
  a dynamic Bayesian network for genomic segmentation. In preparation.

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

Using :option:`--num-starts`\=\ *starts* will generate multiple copies of the
``input.master`` file, named ``input.0.master``, ``input.1.master``,
and so on, with different randomly picked initial parameters. You may
substitute your own ``input.master`` files but I recommend starting
with a Segway-generated template. This will help avoid some common
pitfalls. In particular, if you are going to perform training on your
model, you must ensure that the ``input.master`` file retains the same
``#ifdef`` structure for parameters you wish to train. Otherwise, the
values discovered after one round of training will not be used in
subsequent rounds, or in the identify or posterior stages.

You can use the :option:`--num-labels`\=\ *labels* option to specify the
number of segment labels to use in the model (default 2). You can set
this to a single number or a range with Python slice notation. For
example, ``--num-labels=5:20:5`` will result in 5, 10, and 15 labels
being tried. If you specify :option:`--num-starts`\=\ *starts*, then
there will be *starts* different threads for each of the *labels*
labels tried.

Segway allows multiple models of the values of an observation track
using three different probability distributions: the normal
distribution, the gamma distribution, and a multinomial distribution,
where the nominal output classes each map to a different bin of the
numerical data.

XXX cleanup duplication

The model may be generated using a normal distribution for continuous
observed tracks (``--distribution=norm``, the default), a normal
distribution on asinh-transformed data
(``--distribution=asinh_norm``), or a gamma distribution
(``--distribution=gamma``). The ideal methodology for setting gamma
parameter values is less well-understood, and it also requires an
unreleased version of GMTK. I recommend the use of ``asinh_norm`` in
most cases.

For gamma distributions, Segway generates initial parameters by
converting mean~$\mu$ and variance~$\sigma^2$ to shape~$k$ and
scale~$\theta$ using the equations~$\mu = k \theta$ and~$\sigma^2 = k
\theta^2$.

XXX add arcsinh_normal similar to log

You may specify a subset of tracks using the ``--trackname`` option
which may be repeated. For example::

    segway --trackname dnasei --trackname h3k36me3

will include the two tracks ``dnasei`` and ``h3k36me3`` and no others.

It is very important that you always specify the same ``--trackname``
options at all stages in the Segway workflow. There is also a special
track name, ``dinucleotide``. When you specify
``--trackname=dinucleotide``, Segway will create a track containing
the dinucleotide that starts at a particular position. This can help
in modeling CpG or G+C bias.

Segment length constraints
==========================

The XXX option allows specification of minimum and maximum segment
lengths for various labels. XXX include sample of table

also a way to add a soft prior on XXX cover --prior-strength
XXX default is XXXcomp, this can't be changed at the moment. E-mail
Michael if you need it to be changable.

Distributed computing
=====================
Segway can currently perform training and identification tasks only
using a cluster controllable with the DRMAA (cite) interface. I have
only tested it against Sun Grid Engine, but it should be possible to
work with other DRMAA-compatible distriuted computing systems, such as
Platform LSF, PBS, Condor, (XXXcomp add others). If you are interested
in using one of these systems, please contact me so we can correct all
the fine details. A standalone version is planned.

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
limited studies provide data just for these regions. There is a file
containing only nine of these regions at XXXcomp(make it), which
covers 0.15% of the human genome, and is useful for training.

Memory usage
============

XXX describe new regime

other sections of workflow XXX

XXX add section on all other options

Command-line usage summary
==========================

XXX cover all of these options.

::

  Usage: segway [OPTION]... GENOMEDATADIR
  
  Options:
    --version             show program's version number and exit
    -h, --help            show this help message and exit
  
    Data subset:
      -t TRACK, --track=TRACK
                          append TRACK to list of tracks to use (default all)
      --include-coords=FILE
                          limit to genomic coordinates in FILE
      --exclude-coords=FILE
                          filter out genomic coordinates in FILE
  
    Model files:
      -i FILE, --input-master=FILE
                          use or create input master in FILE
      -s FILE, --structure=FILE
                          use or create structure in FILE
      -p FILE, --trainable-params=FILE
                          use or create trainable parameters in FILE
      --dont-train=FILE   use FILE as list of parameters not to train
      --seg-table=FILE    load segment hyperparameters from FILE
      --semisupervised=FILE
                          semisupervised segmentation with labels in FILE
  
    Output files:
      -b FILE, --bed=FILE
                          create bed track in FILE
  
    Intermediate files:
      -o DIR, --observations=DIR
                          use or create observations in DIR
      -d DIR, --directory=DIR
                          create all other files in DIR
  
    Variables:
      -D DIST, --distribution=DIST
                          use DIST distribution
      -r NUM, --random-starts=NUM
                          randomize start parameters NUM times (default 1)
      -N SLICE, --num-segs=SLICE
                          make SLICE segment classes (default 2)
      --prior-strength=RATIO
                          use RATIO times the number of data counts as the
                          number of pseudocounts for the segment length prior
                          (default 0)
      -m PROGRESSION, --mem-usage=PROGRESSION
                          try each float in PROGRESSION as the number of
                          gibibytes of memory to allocate in turn (default
                          2,3,4,6,8,10,12,14,15)
      -v NUM, --verbosity=NUM
                          show messages with verbosity NUM
      --drm-opt=OPT       specify an option to be passed to the distributed
                          resource manager
  
    Flags:
      -c, --clobber       delete any preexisting files
      -T, --no-train      do not train model
      -I, --no-identify   do not identify segments
      -P, --no-posterior  do not identify probability of segments
      -k, --keep-going    keep going in some threads even when you have errors
                          in another
      -n, --dry-run       write all files, but do not run any executables
      -S, --split-sequences
                          split up sequences that are too large to fit into
                          memory
  
  

Python interface
================
I have designed Segway such that eventually one may call different
components directly from within Python. To do so, import the following
module:

XXXcomp table here (from the setup.py)

You can then call the appropriate module through its ``main()``
function with the same arguments you would use at the command line.
For example::

  from segway import run

  GENOMEDATA_DIRNAME = "genomedata"

  run.main("--no-identify", GENOMEDATA_DIRNAME)

All other interfaces (the ones that do not use a ``main()`` function)
to Segway code are undocumented and should not be used. If you do use
them,know that the API may change at any time without notice.
