======================
 Segway documentation
======================
:Author: Michael M. Hoffman <mmh1 at uw dot edu>
:Organization: University of Washington
:Address: Department of Genome Sciences, PO Box 355065, Seattle, WA 98195-5065, United States of America
:Copyright: 2009 Michael M. Hoffman

.. include:: <isogrk3.txt>

For a conceptual overview see the paper:

  Michael M. Hoffman, Orion Buske, Jeff A. Bilmes, William Stafford
  Noble. Segway: a dynamic Bayesian network for genomic segmentation.
  In preparation.

Michael <mmh1 at uw dot edu> can send you a copy of the latest
manuscript.

The workflow
============
Segway requires genomic data in ``genomedata`` format, and with a
single command-line will--

  1. generate an unsupervised segmentation model and initial
     parameters appropriate for this data;
  2. **train** parameters of the model starting with the initial parameters;
  3. **identify** segments in this data with the model; and
  4. calculate **posterior** probability for each possible segment label
     at each position.

It is also possible to run each of these steps independently. For
example, you may want to specify your own model (including models for
tasks quite unlike segmentation), train it on one dataset, and then
perform posterior decoding on another dataset, skipping the identify
stage. This is easily possible.

Technical description
---------------------
More specifically, Segway performs the following steps:

  1. Acquires data in ``genomedata`` format
  2. Generates an appropriate model for unsupervised
     segmentation (``segway.str``, ``segway.inc``) for use by GMTK
  3. Generates appropriate initial parameters (``input.master``
     or ``input.*.master``) for use by GMTK
  4. Writes the data in a format usable by GMTK
  5. Call GMTK to perform expectation maximization (EM) training,
     resulting in a parameter file (``params.params``)
  6. Call GMTK to perform Viterbi decoding of the observations
     using the generated model and discovered parameters
  7. Convert the GMTK Viterbi results into BED format
     (``segway.bed.gz``) for use in a genome browser, or by
     ``segtools``, or other tools
  8. Call GMTK to perform posterior decoding of the observations
     using the generated model and discovered parameters
  9. Convert the GMTK posterior results into wiggle format
     (``posterior.seg*.wig.gz``) for use in a genome browser or
     other tools
  10. Use a distributed computing system to parallelize all of the
      GMTK tasks listed above, and track and predict their resource
      consumption to maximize efficiency
  11. Generate reports on the established likelihood at each round of
      training (``likelihood.*.tab``)
  12. (not implemented) Call ``segtools`` for a more comprehensive report
      and plots on the resulting segmentation

Generating the model
====================

Segway generates a model (``segway.str``) and initial parameters
(``input.master``) appropriate to a dataset using the GMTKL
specification language and the GMTK master parameter file format. Both
of these are described more fully in the GMTK documentation (cite),
and the default structure and starting parameters are described more
fully in the Segway article.

You can tell Segway just to generate these files and not to perform
any inference using the :option:`--dry-run` option.

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

Segway allows multiple models of the values of a continuous
observation tracks using three different probability distributions: a
normal distribution (``--distribution=norm``), a normal distribution
on asinh-transformed data (``--distribution=asinh_norm``, the
default), or a gamma distribution (``--distribution=gamma``). For
gamma distributions, Segway generates initial parameters by converting
mean |mu| and variance |sigma|:sup:`2` to shape *k* and scale \theta
using the equations |mu| = *k*\ |theta| and |sigma|:sup:`2` =
*k*\ |theta|:sup:`2`. The ideal methodology for setting gamma parameter
values is less well-understood, and it also requires an unreleased
version of GMTK. I recommend the use of ``asinh_norm`` in most cases.

You may specify a subset of tracks using the :option:`--trackname` option
which may be repeated. For example::

    segway --trackname dnasei --trackname h3k36me3

will include the two tracks ``dnasei`` and ``h3k36me3`` and no others.

It is very important that you always specify the same
:option:`--trackname` options at all stages in the Segway workflow.
There is also a special track name, ``dinucleotide``. When you specify
``--trackname=dinucleotide``, Segway will create a track containing
the dinucleotide that starts at a particular position. This can help
in modeling CpG or G+C bias.

Segment length constraints
==========================

The XXX option allows specification of minimum and maximum segment
lengths for various labels. XXX include sample of table

also a way to add a soft prior on XXX cover --prior-strength
XXX default is XXXcomp, this can't be changed at the moment.

Distributed computing
=====================
Segway can currently perform training and identification tasks only
using a cluster controllable with the DRMAA (cite) interface. I have
only tested it against Sun Grid Engine and Platform LSF, but it should
be possible to work with other DRMAA-compatible distributed computing
systems, such as PBS Pro, PBS/TORQUE, Condor, or GridWay. If you are
interested in using one of these systems, please contact Michael so he
correct all the fine details. A standalone version is planned.

Training
========
Most users will generate the model at training time, but to specify
your own model there are the :option:`--structure`\=\ *filename* and
:option:`--input-master`\=\ *filename* options.

Training can be a time-consuming process. You may wish to train only
on a subset of your data. To facilitate this, there is an
:option:`--input-regions`\=\ *filename* option which specifies a BED
file containing a list of regions to limit to. For example, the ENCODE
Data Coordination Center at University of California Santa Cruz keeps
the coordinates of the ENCODE pilot regions in this format at
<http://hgdownload.cse.ucsc.edu/goldenPath/hg18/database/encodeRegions.txt.gz>.
For human whole-genome studies, these regions have nice properties
since they mark 1 percent of the genome, and were carefully picked to
include a variety of different gene densities, and a number of more
limited studies provide data just for these regions. There is a file
containing only nine of these regions at
<http://noble.gs.washington.edu/~mmh1/software/segway/data/regions.manual.1.tab>,
which covers 0.15% of the human genome, and is useful for training.
All coordinates are in terms of the NCBI36 assembly of the human
reference genome (also called ``hg18`` by UCSC).

Memory usage
============

Inference on complex models or long sequences can be memory-intensive.
In order to work efficiently when it is not always easy to predict
memory use in advance, Segway controls the memory use of its subtasks
on a cluster with a trial-and-error approach. It will submit jobs to
your clustering system specifying the amount of memory they are
expected to take up. Your clustering system will allocate these jobs
such that they XXX. If a job takes up more memory than allocated, then
it will be killed and restarted with a larger amount of memory
allocated, along the progression specified in gibibytes by
:option:`--mem-usage`\=\ *progression*. The default *progression* is
2,3,4,6,8,10,12,14,15.

XXX memory use XXX also, in current version of GMTK, there is a
problem with running out of dynamic range on sequences that are too
large that manifests itself as a "zero clique error." This will be
fixed in GMTK soon. XXX :option:`--split-sequences`\=\ *size* XXX default
2,000,000.

XXX new sections: other sections of workflow

XXX new sections: other sections of technical description

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
      --resolution=RES    downsample to every RES bp (default 1)
      --ruler-scale=SCALE
                          ruler marking every SCALE bp (default 10)
      --prior-strength=RATIO
                          use RATIO times the number of data counts as the
                          number of pseudocounts for the segment length prior
                          (default 0)
      -m PROGRESSION, --mem-usage=PROGRESSION
                          try each float in PROGRESSION as the number of
                          gibibytes of memory to allocate in turn (default
                          2,3,4,6,8,10,12,14,15)
      -S SIZE, --split-sequences=SIZE
                          split up sequences that are larger than SIZE bp
                          (default 2000000)
      -v NUM, --verbosity=NUM
                          show messages with verbosity NUM
      --cluster-opt=OPT   specify an option to be passed to the cluster manager
  
    Flags:
      -c, --clobber       delete any preexisting files
      -T, --no-train      do not train model
      -I, --no-identify   do not identify segments
      -P, --no-posterior  do not identify probability of segments
      -k, --keep-going    keep going in some threads even when you have errors
                          in another
      -n, --dry-run       write all files, but do not run any executables


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

XXX describe runner.fromoptions() interface

All other interfaces (the ones that do not use a ``main()`` function)
to Segway code are undocumented and should not be used. If you do use
them, know that the API may change at any time without notice.

Support
=======

For support of Segway, please write to the <segway-users@uw.edu> mailing
list, rather than writing the authors directly. Using the mailing list
will get your question answered more quickly. It also allows us to
pool knowledge and reduce getting the same inquiries over and over.
You can subscribe here:

https://mailman1.u.washington.edu/mailman/listinfo/segway-users

Specifically, if you want to report a bug or request a feature, please
do so using the Segway issue tracker at:

http://code.google.com/p/segway-genome/issues/

If you do not want to read discussions about other people's use of
Segway, but would like to hear about new releases and other important
information, please subscribe to <segway-announce@uw.edu> by visiting
this web page:

https://mailman1.u.washington.edu/mailman/listinfo/segway-announce

Announcements of this nature are sent to both `segway-users` and
`segway-announce`.
