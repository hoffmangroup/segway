==========================
 Segway |version| Overview
==========================

.. include:: <isogrk3.txt>

Installation
============

Segway requires the following prerequisites:

You need Python 2.6 or 2.7.

You need Graphical Models Toolkit (GMTK), which you can get at
<http://melodi.ee.washington.edu/downloads/gmtk/gmtk-1.0.1.tar.gz>.

You need Genomedata 1.3.5 or later. To install Genomedata see the instructions
at <http://pmgenomics.ca/hoffmanlab/proj/genomedata/>.

Afterwards Segway can be installed automatically with the command ``pip install
segway``.


Standalone configuration
------------------------
Segway can be run without any cluster system. This will automatically be
used when Segway fails to access any cluster system. You can force it by
setting the `SEGWAY_CLUSTER` environment variable to `local`. For example,
if you are using bash as your shell, you can run:

    SEGWAY_CLUSTER=local segway

Cluster configuration
---------------------
If you want to use Segway with your cluster, you will need the
``drmaa>=0.4a3`` Python package.

You need either Sun Grid Engine (SGE; now called Oracle Grid Engine),
Platform Load Sharing Facility (LSF) and FedStage DRMAA for LSF, or
Torque/PBS/PBS Pro (experimental).

If FedStage DRMAA for LSF is installed, Segway should be ready to go
on LSF out of the box.

If you are using SGE, someone with cluster manager privileges on your
cluster must have Segway installed within their PYTHONPATH or
systemwide and then run ``python -m segway.cluster.sge_setup``. This
sets up a consumable mem_requested attribute for every host on your
cluster for more efficient memory use.

The workflow
============
Segway accomplishes four major tasks from a single command-line. It--

  1. **generates** an unsupervised segmentation model and initial
     parameters appropriate for this data;
  2. **trains** parameters of the model starting with the initial
     parameters; and
  3. **identifies** segments in this data with the model.
  4. calculates **posterior** probability for each possible segment
     label at each position.

.. todo: block diagram

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
     Segtools <http://pmgenomics.ca/hoffmanlab/proj/segtools/>, or other tools

  8. Call GMTK to perform posterior decoding of the observations
     using the generated model and discovered parameters
  9. Convert the GMTK posterior results into bedGraph format
     (``posterior.seg*.bedGraph.gz``) for use in a genome browser or
     other tools
  10. Use a distributed computing system to parallelize all of the
      GMTK tasks listed above, and track and predict their resource
      consumption to maximize efficiency
  11. Generate reports on the established likelihood at each round of
      training (``likelihood.*.tab``)

..  12. (not implemented) Call Segtools
..      <http://pmgenomics.ca/hoffmanlab/proj/segtools/> for a more
..      comprehensive report and plots on the resulting segmentation.

The **identify** and **posterior** tasks can run simultaneously, as
they depend only on the results of **train**, and not each other.

Data selection
==============
Segway accepts data only in the Genomedata format. The Genomedata
package includes utilities to convert from BED, wiggle, and bedGraph
formats. By default, Segway uses all the continuous data tracks in a
Genomedata archive.

Tracks
------

You may specify a subset of tracks using the :option:`--track` option
which may be repeated. For example::

    segway --track dnasei --track h3k36me3

will include the two tracks ``dnasei`` and ``h3k36me3`` and no others.

It is very important that you always specify the same
:option:`--track` options at all stages in the Segway workflow.
There is also a special track name, ``dinucleotide``. When you specify
``--track=dinucleotide``, Segway will create a track containing
the dinucleotide that starts at a particular position. This can help
in modeling CpG or G+C bias.

You can run a concatenated segmentation by separating tracks with a
comma. For example::

    segway --track dnasei.liver,dnasei.blood --track h3k36me3.liver,h3k36me3.blood

.. _positions:

Positions
---------
By default, Segway runs analyses on the whole genome. This can be
incredibly time-consuming, especially for training. In reality,
training (and even identification) on a smaller proportion of the
genome is often sufficient. There are also regions as the genome such
as those containing many repetitive sequences, which can cause
artifacts in the training process. The :option:`--exclude-coords`\=\
*file* and :option:`--include-coords`\=\ *file* options specify BED
files with regions that should be excluded or included respectively.
If both are specified, then inclusions are prcoessed first and the
exclusions are then taken out of the included regions.

.. note::

  BED format uses zero-based half-open coordinates, just like
  Python. This means that the first nucleotide on chromosome 1 is
  specified as::

      chr1    0    1

  The UCSC Genome Browser and Ensembl web interfaces, as well as the
  wiggle formats use the one-based fully-closed convention, where it is
  called *chr1:1-1*.

For example, the ENCODE Data Coordination Center at University of
California Santa Cruz keeps the coordinates of the ENCODE pilot
regions in this format at
<http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/referenceSequences/encodePilotRegions.hg19.bed>
(GRCh37/hg19) and
<http://hgdownload.cse.ucsc.edu/goldenPath/hg18/database/encodeRegions.txt.gz>
(NCBI36/hg18). For human whole-genome studies, these regions have nice
properties since they mark 1 percent of the genome, and were carefully
picked to include a variety of different gene densities, and a number
of more limited studies provide data just for these regions. All
coordinates are in terms of the GRCh37 assembly of the human reference
genome (also called ``hg19`` by UCSC).


After reading in data from a Genomedata archive, and selecting a
smaller subset with :option:`--exclude-coords` and
:option:`--include-coords`, the final included regions are referred to
as *windows*, and are supplied to GMTK for inference. There is no
direction connection between the data in different windows during any
inference process--the windows are treated independently.

Resolution
----------

.. important::

  The resolution feature is not implemented in the current Segway
  |version| release.

In order to speed up the inference process, you may downsample the
data to a different resolution using the :option:`--resolution`\=\
*res* option. This means that Segway will partition the input
observations into fixed windows of size *res* and perform inference on
the mean of the observation averaged along the fixed window. This can
result in a large speedup at the cost of losing the highest possible
precision. However, if you are considering data that is only generated
at a low resolution to start with, this can be an appealing option.

.. warning::

  You must use the same resolution for *both* training and
  identification.

Model generation
================

Segway generates a model (``segway.str``) and initial parameters
(``input.master``) appropriate to a dataset using the GMTKL
specification language and the GMTK master parameter file format. Both
of these are described more fully in the GMTK documentation (cite),
and the default structure and starting parameters are described more
fully in the Segway article.

The starting parameters are generated using data from the whole
genome, which can be quickly found in the Genomedata archive. Even if
you are training on a subset of the genome, this information is not
used.

You can tell Segway just to generate these files and not to perform
any inference using the :option:`--dry-run` option.

Using :option:`--num-instances`\=\ *starts* will generate multiple copies of the
``input.master`` file, named ``input.0.master``, ``input.1.master``,
and so on, with different randomly picked initial parameters. Segway
training results can be quite dependent on the initial parameters
selected, so it is a good idea to try more than one. I usually use
`--num-instances=10``.

You may substitute your own ``input.master`` files but I recommend
starting with a Segway-generated template. This will help avoid some
common pitfalls. In particular, if you are going to perform training
on your model, you must ensure that the ``input.master`` file retains
the same ``#ifdef`` structure for parameters you wish to train.
Otherwise, the values discovered after one round of training will not
be used in subsequent rounds, or in the identify or posterior stages.

You can use the :option:`--num-labels`\=\ *labels* option to specify the
number of segment labels to use in the model (default 2). You can set
this to a single number or a range with Python slice notation. For
example, ``--num-labels=5:20:5`` will result in 5, 10, and 15 labels
being tried. If you specify :option:`--num-instances`\=\ *starts*, then
there will be *starts* different instances for each of the *labels*
labels tried.

The question of finding the right number of labels is a difficult one.
Mathematical criteria, such as the Bayesian information criterion,
would usually suggest using higher numbers of labels. However, the
results are difficult for a human to interpret in this case. This is
why we usually use ~25 labels for a segmentation of dozens of input
tracks. If you use a small number of input tracks you can probably use
a smaller number of labels.

There is an experimental :option:`--num-sublabels`\=\ *sublabels*
option that enables hierarchical segmentation, where each segment
label is divided into a number of segment sublabels, each one with its
own Gaussian emission parameters. The
output segmentation will be defined according to the 
:option:`--output-label`\=\ *output_label* option, by default *seg*, 
which will output by (super) segment label as normal. *subseg* 
will output in terms of individual sublabels, only printing out the 
sublabel part, and *full* will print out both the superlabel and
the sublabel, separated by a period. For example, a coordinate
assigned superlabel 1 and sublabel 0 would display as "1.0".
Using this feature effectively may require manipulation of model
parameters.

Segway allows multiple models of the values of a continuous
observation tracks using three different probability distributions: a
normal distribution (``--distribution=norm``), a normal distribution
on asinh-transformed data (``--distribution=asinh_norm``, the
default), or a gamma distribution (``--distribution=gamma``). For
gamma distributions, Segway generates initial parameters by converting
mean |mu| and variance |sigma|:sup:`2` to shape *k* and scale \theta
using the equations |mu| = *k*\ |theta| and |sigma|:sup:`2` = *k*\
|theta|:sup:`2`. The ideal methodology for setting gamma parameter
values is less well-understood. I recommend the use of ``asinh_norm``
in most cases.

.. _segment-duration-model:

Segment duration model
----------------------

.. _hard-length-constraints:

Hard length constraints
~~~~~~~~~~~~~~~~~~~~~~~

The :option:`--seg-table`\=\ *file* option allows specification of a
*segment table* that specifies minimum and maximum segment lengths for
various labels. Here is an example of a segment table::

  label	len
  1:4	200:2200:50
  0	200::50
  4:	200::50

The file is tab-delimited, and the header line with ``label`` in one
column and ``len`` in another is mandatory. The first column specifies
a label or range of labels to which the constraints apply.  In this
column, a range of label values may be specified using Python slice
syntax, so label ``1:4`` specifies labels 1, 2, and 3.  Using ``4:``
for a label, as in the last row above, means all labels 4 or higher.

The second column specifies three colon-separate values: the minimum
segment length, maximum segment length, and the ruler.  In the example
above, for labels 1, 2 and 3, segment lengths between 200 and 2200 are
allowed, with a 200 bp ruler. If either the minimum or maximum lengths
are left unspecified, then no corresponding constraint is applied.

The ruler is an efficient heuristic that decreases the memory used
during inference at the cost of also decreasing the precision with
which the segment duration model acts.  Essentially, it allows the
duration model to switch the behavior of the rest of the model only
after a multiple of *scale* bp has passed.  Note that the ruler for
every label must be explicitly specified and must match all other
ruler entries in this file, as well as the option set with
:option:`--ruler-scale`\=\ *scale*. (This may become more free in the
future.)

Due to the lack of an epilogue in the model, it is possible to get one
segment per sequence that actually does not meet the minimum segment
length. This is expected and will be fixed in a future release.

Note that increasing hard minimum or maximum length constraints will
greatly increase memory use and run time. You can decrease this
performance penalty by increasing ruler size (which makes the
precision of the duration model a little coarser), or by using the
soft length priors below.

Use these segment lengths along with the supervised learning feature
with caution. If you try to create something impossible with your
supervision labels, such as defining a 2300-bp region to have label 1,
which you have already constrained to have a maximum segment length of
2200, GMTK will produce the dreaded zero clique error and your
training run will fail. Don't do this. In practice, due to the
imprecision introduced by the 200-bp ruler, a region labeled in the
supervision process with label 1 that is only 2000 bp long may also
cause the training process to fail with a zero clique error. If this
happens either decrease the size of the ruler, increase the size of
the maximum segment length, or decrease the size of the supervision
region.

Soft length prior
~~~~~~~~~~~~~~~~~

There is also a way to add a soft prior on the length distribution,
which will tend to make the expected segment length 100000, but will
still allow data that strongly suggests another length control. The
default expected segment length of 100000 can't be changed at the
moment but will in a future version.

You can control the strength of the prior relative to observed
transitions with the :option:`--prior-strength`\=\ *strength* option.
Setting ``--prior-strength=1`` means there are as many pseudocounts
due to the prior as the number of nucleotides in the training regions.

The :option:`--segtransition-weight-scale`\=\ *scale* option controls
the strength of the prior in another way. It controls the strength of
the length prior relative to the data from the observed tracks. The
default *scale* of 1 gives the soft transition model equal strength to
a single data track. Using higher or lower values gives comparatively
greater or lesser weight to the probability from the soft length
prior, essentially allowing the prior to have more votes in
determining where a segment boundary is. The impact of the prior will
be a function of both :option:`--segtransition-weight-scale` and
:option:`--prior-strength`.

One may effectively use the hard length constraints and soft length
priors simultaneously.

Task selection
==============
Segway will perform either (a) model generation and training or (b)
identification separately, so it is possible to train on a subset of
the genome and identify on the whole thing. To train, use::

  segway train GENOMEDATA TRAINDIR

To identify, specify the TRAINDIR you used in the first round::

  segway identify GENOMEDATA TRAINDIR IDENTIFYDIR

In both cases, replace GENOMEDATA with the Genomedata archive you're
using. The use of :option:`--dry-run` will cause Segway to generate
appropriate model and observation files but not to actually perform
any inference or queue any jobs. This can be useful when
troubleshooting a model or task.

Train task
==========
Most users will generate the model at training time, but to specify
your own model there are the :option:`--structure`\=\ *filename* and
:option:`--input-master`\=\ *filename* options. You can simultaneously
run multiple *instances* of EM training in parallel, specified with the
:option:`--instances`\=\ *instances* option. Each instance consists of
a number of rounds, which are broken down into individual tasks for
each training region. The results from each region for a particular
instance and round are combined in a quick *bundle* task. It results in
the generation of a parameter file like ``params.3.params.18`` where
``3`` is the instance index and ``18`` is the round index. Training for
a particular instance continues until at least one of these criteria is
met:

* the likelihood from one round is only a small improvement from the
  previous round; or
* 100 rounds have completed.

Specifically, the "small improvement" is defined in terms of the
current likelihood :math:`L_n` and the log likelihood from the
previous round :math:`L_{n-1}`, such that training continues while

.. math::

   \left| \dfrac{\log L_n - \log L_{n-1}}{\log L_{n-1}} \right| \ge 10^{-5}.

This constant will likely become an option in a future version of
Segway.

As EM training produces diminishing returns over time, it is likely
that one can obtain acceptably trained parameters well before these
criteria are met. Training can be a time-consuming process. You may
wish to train only on a subset of your data, as described in
:ref:`positions`.

When all instances are complete, Segway picks the parameter set with the
best likelihood and copies it to ``params.params``.

There are two different modes of training available, unsupervised and
semisupervised.

Unsupervised training
---------------------
By default, Segway trains in unsupervised mode, which is a form of
clustering. In this mode, it tries to find recurring patterns
suggested by the data without any additional preconceptions of which
regions should be tied together.

Semisupervised training
-----------------------
Using the :option:`--semisupervised`\=\ *file* option, one can specify
a BED file as a list of regions used s supervision labels. The *name*
field of the BED File specifies a label to be enforced during
training. For example, with the line::

    chr3    400    800   2

one can enforce that those positions will have label 2. You might do
this if you had specific reason to believe that these regions were
enhancers and wanted to find similar patterns in your data tracks.
Using smaller labels first (such as 0) is probably better. Supervision
labels are not enforced during the identify task, and therefore cannot
be specified during identify.

To simulate fully supervised training, simply supply supervision
labels for the entire training region.

None of the supervision labels can overlap with each other. You should
combine any overlapping labels before specifying them to Segway.

General options
---------------
The :option:`--dont-train`\=\ *file* option specifies a file with a
newline-delimited list of parameters not to train. By default, this
includes the ``dpmf_always``, ``start_seg``, and all GMTK
DeterministicCPT parameters. You are unlikely to use this unless you
are generating your own models manually.

Seeding
~~~~~~~
Segway can be forced to run with a specified random number generator seed by
setting the `SEGWAY_RAND_SEED` environment variable. This can be useful for
reproducing results in the future. For example, if you are using bash as your
shell you can run:

    SEGWAY_RAND_SEED=1498730685

To set the random number generator seed to the number 1498730685.

Recovery
--------
Since training can take a long time, this increases the probability
that external factors such as a system failure will cause a training
run to fail before completion. You can use the :option:`--recover`\=\
*dirname* option to specify a previous work directory you're
recovering from.

Identify task
=============

The **identify** mode of Segway uses the Viterbi algorithm to decode
the most likely path of segments, given data and a set of parameters,
which can come from the **train** task. Identify runs considerably
more quickly than training. While the underlying inference task is
very similar, it must be completed on each region of interest only
once rather than hundreds of times as in training.

You can either manually set individual input master, parameter, and
structure files, or implicitly use the files generated by the
**train** task completed in *traindir*, and referenced in *traindir*\
``/train.tab``. If you are using training data from an old version of
Segway, you must either create a ``train.tab`` file or specify the
parameters manually, using :option:`--structure`,
:option:`--input-master`, and :option:`--trainable-params`.

The :option:`--bed`\=\ *bedfile* option specifies where the
segmentation should go. If *bedfile* ends in ``.gz``, then Segway uses
gzip compression. The default is ``segway.bed.gz`` in the working
directory.

You can load the generated BED files into a genome browser. Because
the files can be large, I recommend placing them on your web site and
supplying the URL to the genome browser rather than uploading the file
directly. When using the UCSC genome browser, the bigBed utility may
be helpful in speeding access to parts of a segmentation.

The output is in BED format
(http://genome.ucsc.edu/FAQ/FAQformat.html#format1), and includes
columns for chromosome, start, and end (in zero-based, half-open
format), and the label. Other columns to the right are used for
display purposes, such as coloring the browser display, and can be
safely ignored for further processing. We use colors from ColorBrewer
(http://colorbrewer2.org/).

Recovery
--------
The :option:`--recover`\=\ *dirname* allows recovery from an
interrupted identify task. Segway will requeue jobs that never
completed before, skipping any windows that have already completed.

Creating layered output
-----------------------
Segway produces BED files as output with the segment label in the name
field. While this is the most sensible way of interchanging the
segmentation with other programs, it can be difficult to visualize. To
solve this problem, Segway will also produce a *layered* BED file with
rows for each possible Segment label and thick boxes at the location
of each label. This is what we show in the screenshot figure of the
Segway article. This is much easier to see at low levels of
magnification. The layers are also labeled, removing the need to
distinguish them exclusively by color. While Segway automatically
creates these files at the end of an identify task, you can also use
:program:`segway-layer` wit ha standard BED segmentation to repeat the
layering process, which you may want to do if you want to add mnemonic
labels instead of the initial integers used as labels.
:program:`segway-layer` supports the use of standard input and output
by using ``-`` as a filename, following a common Unix convention.

The mnemonic files used by Segway and Segtools have a simple format.
They are tab-delimited files with a header that has the following
columns: ``old``, ``new``, and ``description``. The ``old`` column
specifies the original label in the BED file, which is always produced
as an integer by Segway. The ``new`` column allows the specification
of a short alphanumeric mnemonic for the label. The ``description``
column is unused by :program:`segway-layer`, but you can use it to add
helpful annotations for humans examining the list of labels, or to
save label mnemonics you used previously. The row order of the
mnemonic file matters, as the layers will be laid down in a similar
order. Mnemonics sharing the same alphabetical prefix (for example,
``A0`` and ``A1``) or characters before a period (for example, ``0.0``
and ``0.1``) will be rendered with the same color.

:program:`segtools-gmtk-parameters` in the Segtools package can
automatically identify an initial hierarchical labeling of
segmentation parameters.  This can be very useful as a first
approximation of assigning meaning to segment labels.

A simple mnemonic file appears below::

  old	new	description
  0	TSS	transcription start site
  2	GE	gene end
  1	D	dead zone

Posterior task
==============
The **posterior** inference task of Segway estimates for each position
of interest the probability that the model has a particular segment
label given the data. This information is delivered in a series of
numbered wiggle files, one for each segment label. In hierarchical 
segmentation mode, setting the `--output-label` option to *full* or
*subseg* will cause segway to produce a wiggle file for each sublabel
instead, identified using the label and the sublabel in the file name 
before the file extension. For example, the bedGraph file for label 0, 
and sublabel 1 would be called ``posterior0.1.bedGraph``. The individual 
values will vary from 0 to 100, showing the percentage probability at 
each position for the label in that file. In most positions, the value 
will be 0 or 100, and substantially reproduce the Viterbi path 
determined from the **identify** task. The **posterior** task uses the 
same options for specifying a model and parameters as **identify**.

Posterior results can be useful in determining regions of ambiguous
labeling or in diagnosing new models. The mostly binary nature of the
posterior assignments is a consequence of the design of the default
Segway model, and it is possible to design a model that does not have
this feature. Doing so is left as an exercise to the reader.

.. todo: name the files

You may find you need to convert the bedGraph files to bigWig format
first to allow small portions to be uploaded to a genome browser
piecewise.

.. todo: same options for specifying model and parameters as identify

Recovery
--------
Recovery is not yet supported for the posterior task.

Python interface
================
I have designed Segway such that eventually one may call different
components directly from within Python.

You can then call the appropriate module through its ``main()``
function with the same arguments you would use at the command line.
For example::

  from segway import run

  GENOMEDATA_DIRNAME = "genomedata"

  run.main(["--random-starts=3", "train", GENOMEDATA_DIRNAME])

All other interfaces (the ones that do not use a ``main()`` function)
to Segway code are undocumented and should not be used. If you do use
them, know that the API may change at any time without notice.



Command-line usage summary
==========================

All programs in the Segway distribution will report a brief synopsis
of expected arguments and options when the :option:`--help` option is
specified and version information when :option:`--version` is specified.

.. include:: _build/cmdline-help/segway.help.txt

Utilities
---------

.. include:: _build/cmdline-help/segway-layer.help.txt

.. include:: _build/cmdline-help/segway-winner.help.txt

Running Segway for large jobs
=============================

It is highly recommended that a terminal mutliplexer, such as `tmux`_ or `GNU
screen`_, is used to manage your terminal sessions running Segway. Using either
of these programs allows you to create a session to run longer segway jobs that
you can safely detatch from without losing your work. For more information, see
their respective documentation.

.. _tmux: http://tmux.sourceforge.net/
.. _GNU Screen: http://www.gnu.org/software/screen/

Helpful commands
================
Here are some short bash scripts or one-liners that are useful:

There used to be a recipe here to continue Segway from an interrupted
training run, but this has been replaced by the `--old-directory` option.

Make a tarball of parameters and models from various directories::

    (for DIR in traindir1 traindir2; do
    echo $DIR/{auxiliary,params/input.master,params/params.params,segway.str,triangulation}
    done) | xargs tar zcvf training.params.tar.gz

.. todo: include rsync-segway script

Rsync parameters from `$REMOTEDIR` on `$REMOTEHOST` to `$LOCALDIR`::

    rsync -rtvz --exclude output --exclude posterior --exclude viterbi \
    --exclude observations --exclude "*.observations" --exclude accumulators \
    $REMOTEHOST:$REMOTEDIR $LOCALDIR

Print all last likelihoods::

    for X in likelihood.*.tab; \
    do dc -e "8 k $(tail -n 2 $X | cut -f 1 | xargs echo | sed -e 's/-//g') \
    sc sl ll lc - ll / p"; \
    done

.. todo: research BEDTools capability here

Recover as much as possible from an incomplete identification run without
completing it (which can be done with `--old-directory`. Note that this
does not combine adjacent lines of same segment. BEDTools might be able to do
this for you. You will have to create your own header.txt with appropriate
track lines.
