=================
Technical matters
=================

.. include:: <isonum.txt>

Working files
=============

Segway must create a number of working files in order to accomplish
its tasks, and it does this in the directory specified by the required
*workdir* argument. When running training *workdir* is *TRAINDIR*;
when running identification *workdir* is *IDENTIFYDIR*.

The observation files can be quite large, taking up 8 bytes per track per
position and cannot be compressed. As a result they are written out to a
temporary directory on an as-needed basis. This is because otherwise they could
take terabytes for identifying on the whole human genome with dozens of tracks.

You will find a full description of all the working files in the
:ref:`workdir-files` section.

Temporary files
===============
The **identify** and **posterior** tasks create temporary observation
files in directories indicated by the Python `tempfile.gettempdir()`
function, which searches for an appropriate directory as described in
the documentation for `tempfile.tempdir`
<https://docs.python.org/2/library/tempfile.html#tempfile.tempdir>. If
you need to specify that temporary files go into a particular
directory, set the `TMPDIR` environment variable. It is highly recommended that
you ensure that you temporary directory does not reside on a slow storage
medium such as a NFS filesystem. Since many temporary files are created and
deleted this can significantly impact performance.

Distributed computing
=====================
Segway can currently perform the training, identification, and
posterior tasks only using a cluster controllable with the DRMAA
interface. We have only tested it against Sun Grid Engine and Platform
LSF, but it should be possible to work with other DRMAA-compatible
distributed computing systems, such as PBS Pro, PBS/TORQUE, Condor, or
GridWay. If you are interested in using one of these systems, please
open an issue on Github <https://github.com/hoffmangroup/segway/issues/new> to correct all the fine details. A standalone
option exists when you set the `SEGWAY_CLUSTER` environment variable to `local`. Try
installing the free Open Grid Scheduler on your workstation if you want to run Segway
without a full clustering system.

The :option:`--cluster-opt` option allows the specification of native
options to your clustering system --- those options you might pass
to ``qsub`` (SGE) or ``bsub`` (LSF).

.. todo:  comp include SGE and LSF cluster-opt demos

Memory usage
============
Inference on complex models or long sequences can be memory-intensive.
In order to work efficiently when it is not always easy to predict
memory use in advance, Segway controls the memory use of its subtasks
on a cluster with a trial-and-error approach. It will submit jobs to
your clustering system specifying the amount of memory they are
expected to take up. Your clustering system should allocate these jobs
such that the amount of memory on one host is not overcommitted. If a
job takes up more memory than allocated, then it will be killed and
restarted with a larger amount of memory allocated, along the
progression specified in gibibytes by :option:`--mem-usage`\=\
*progression*. The default *progression* is 2,3,4,6,8,10,12,14,15.

We usually train on regions of no more than 2,000,000 frames, where a
single frame contains the number of nucleotides set by the
:option:`--resolution` option. If you use more than that many, GMTK
might run out of dynamic range. This manifests itself as a "zero
clique error." Identify mode rescales probabilities at every frame so
that this is not a problem. However, you will probably want to split
the input sequences somewhat because larger sequences make more
difficult work units (greater memory and run time costs) and thereby
impede efficient parallelization. The :option:`--split-sequences`\=\
*size* option will split up sequences into windows with *size* base pairs
each. The default *size* is 2,000,000. Decreasing to 500,000 will greatly
improve speed at the cost of more artefacts at split boundaries.

Reporting
=========
Segway produces a number of logs of its activity during tasks, which
can be useful for analyzing its performance or troubleshooting. These
are all in the *workdir*/log directory.

Shell scripts
-------------

Segway produces three shell scripts in the log directory that you can
use to replay its subtasks at different levels of abstractions. The
top-level ``segway.sh`` records the command line used to run Segway.
The ``run.sh`` script gives you the GMTK commands called by Segway. A
small number of these are still produced when :option:`--dry-run` is
specified. The ``details.sh`` script contains the exact commands
dispatched by Segway, including wrapper commands that monitor memory
usage, create and delete local temporary files with observation data,
and convert GMTK's output to BED, among other things.

Segway also writes a ``cmdline`` directory in both the ``traindir``
and ``identifydir``. Each instance has its own folder, and for each
job segway queues, a shell script (with the job's name) is written
containing the GMTK command of the queued job.

For example, ``traindir/cmdline/0/emt0.0.0.uuid.sh`` is the shell
script containing the GMTK commands and arguments for job 
``emt0.0.0.uuid`` in instance 0 of training.

Summary reports
---------------

The ``jobs.*.tab`` file contains a tab-delimited file with each job
Segway dispatched for this instance in a different row, reporting on job
identifier (``jobid``), job name (``jobname``), GMTK program (``prog``), number
of segment labels (``num_segs``), number of frames (``num_frames``), maximum
memory usage (``maxvmem``), CPU time (``cpu``) and exit/error status
(``exit_status``). Jobs are written as they are completed. The exit status is
useful for determining whether the job succeeded (status 0) or failed (any
other value, which is sometimes numeric, and sometimes text, depending on the
clustering system used).

The ``likelihood.*.tab`` files each track the progression of
likelihood during a single instance of EM training. The file has a
single column, one for each round of training, which contains the log
likelihood. More positive values are better.

GMTK reports
------------
The ``jt_info.txt`` and ``jt_info.posterior.txt`` files describe how
GMTK builds a junction tree. It is of interest primarily during GMTK
troubleshooting. You are unlikely to use it.

.. _task-output:

Task output
-----------

The ``output`` directory contains the output of the actual GMTK
commands run by Segway. The ``o`` directory contains standard output
and the ``e`` directory contains standard error. If a job fails and
repeats, the output from the new job is appended to the old. The
:option:`--verbosity`\=\ *verbosity* option controls how much
diagnostic information that GMTK writes into these files. The default
and minimum value is ``0``. Raise this value for more information, and
see the GMTK documentation for a description of various levels of
verbosity. Setting :option:`verbosity`\=\ ``30`` can be particularly
helpful in diagnosing model problems. Keep in mind that very high
values (above ``60``) will produce tons of output===maybe
terabytes.

.. warning::

  Running Segway in identify mode with non-zero verbosity is
  currently not supported and may result in errors.

Performance
===========
Some factors that affect compute time and memory requirements:

* the length of the longest region you are training or identifying on
* the number of tracks
* the number of labels

The longest region forms a bottleneck during training because Segway
cannot start the next round of training before all regions in the
previous round are done. So if you specify three regions, one of which
is 10 Mbp long, and the other are 100 kbp, the 10 Mbp region is going
to be a limiting factor. You can use :option:`--split-sequences` (see
above) to put an upper bound on region size.

Names used by Segway
====================

.. _workdir-files:

Workdir files
-------------

Segway expects to be able to create many of these files anew. To avoid
data loss, by default, it will quit if they already exist. If you use
the :option:`--clobber` option, Segway will overwrite the whole
workdir instead.

.. tabularcolumns:: lp{4.5in}

=================================================== =====================================================
 Filename                                            Description
=================================================== =====================================================
``accumulators/``                                   intermediate files used to pass E-step results to
                                                    the M-step of EM training
|rarr| ``acc.``\ \*\ ``.bin``                       accumulator for a particular instance and
                                                    region (reused each round)
``auxiliary/``                                      miscelaneous model files
|rarr| ``dont_train.list``                          defines list of hidden random variables
                                                    that are not trained
|rarr| ``segway.inc``                               C preprocessor (``cpp``) include file
                                                    used in structure
``cmdline``                                         shell scripts containing commandlines/arguments of
                                                    individual GMTK jobs
``cmdline/0,1,.../``                                GMTK job shell scripts for a particular training
                                                    instance(0,1,...)
``cmdline/identify/``                               GMTK job shell scripts for identification
``intermediate``                                    files containing best training filenames per instance
|rarr| ``train_result.``\ \*\ ``.tab``              per instance information containing the filenames of
                                                    the params resulting in the best likelihood
``likelihood/``                                     GMTK's report of the log likelihood for
                                                    the most recent M-step of EM training
|rarr| ``likelihood.``\ \*\ ``.ll``                 contains text of the last log likelihood value for an
                                                    instance. Segway uses this to decide when to
                                                    stop training
|rarr| ``validation.output.``\ \*\ ``.ll``          contains text of the last validation GMTK output for an
                                                    instance
|rarr| ``validation.output.winner.``\ \*\ ``.ll``   contains text of the current best validation GMTK output
                                                    for an instance
|rarr| ``validation.sum.``\ \*\ ``.ll``             contains text of the last validation set log likelihood
                                                    for an instance
|rarr| ``validation.sum.winner.``\ \*\ ``.ll``      contains text of the current best validation set log 
                                                    likelihood for an instance
``log/``                                            diagnostic information
|rarr| ``details.sh``                               script file that includes the exact
                                                    command-lines queued by Segway, with wrapper scripts
|rarr| ``jobs.``\ \*\ ``.tab``                        tab-delimeted summary of jobs queued per instance,
                                                    including resource informatoin and exit status
|rarr| ``jt_info.txt``                              log file used by GMTK when creating a junction tree
|rarr| ``jt_info.posterior.txt``                    log file used by GMTK when creating a junction tree
                                                    in posterior mode
|rarr| ``likelihood.``\ \*\ ``.tab``                tab-delimited summary of log likelihood by training
                                                    instance; can be used to examine  how fast
                                                    training converges
|rarr| ``validation.sum.``\ \*\ ``.tab``            tab-delimited summary of full validation set log likelihood
                                                    by training instance
|rarr| ``validation.output.``\ \*\ ``.tab``         tab-delimited summary of validation log likelihood for 
                                                    each window in the validation set by training instance
|rarr| ``run.sh``                                   list of commands run by Segway, not
                                                    including wrappers
                                                    that create and clean up temporary files such as
                                                    observations used during identification
|rarr| ``segway.sh``                                reports the command-line used to run Segway itself
|rarr| \*\ ``.``\ \*\ ``.float32``                  continuous data for a particular region
|rarr| \*\ ``.``\ \*\ ``.int``                      indicator data (present/absent) for a particular
                                                    region
|rarr| ``float32.list``                             list of continuous data files
|rarr| ``int.list``                                 list of indicator data files
``output/``                                         diagnostic output of individual GMTK jobs
``output/e/``                                       stderr
``output/e/0,1,``...                                stderr for a particular training instance (0, 1,
                                                    ...)
``output/e/identify``                               stderr for identification
``output/o/``                                       stdout
``params/``                                         generated and trained parameters for a given instance
|rarr| ``input.``\ \*\ ``.master``                  generated hyperparameters and starting parameters
|rarr| ``input.master``                             best set of hyperparameters and starting parameters
|rarr| ``params.``\ \*\ ``.params.``\ \*            trained parameters for a given instance and round
|rarr| ``params.``\ \*\ ``.params``                 final trained parameters for a given instance
|rarr| ``params.params``                            best final set of trained parameters
``segway.bed.gz``                                   segmentation in BED format
``segway.str``                                      dynamic Bayesian network structure
``train.tab``                                       important file locations and hyperparameters used in training, to be passed to identify
``triangulation/``                                  triangulation files used for DBN interface
|rarr| ``segway.str.``\ \*\ ``.``\ \*\ ``.trifile`` triangulation file
``viterbi/``                                        intermediate BED files created during distributed Viterbi decoding,
                                                    which get merged into ``segway.bed.gz``
``window.bed``                                      a BED file containing chromosome regions and the indicies Segway assigns to them
=================================================== =====================================================

Job names
---------

In order to watch Segway's progress on your cluster, it is helpful to
understand how it names jobs. A job name for the training task might
look like this::

  emt0.1.34.traindir.ed03201cea2047399d4cbcc4b62f9827

In this example, ``emt`` means expectation maximization training, the
``0`` means instance 0, the ``1`` means round 1, and the ``34`` means
window 34. The name of the training directory is ``traindir``, and
``ed03201cea2047399d4cbcc4b62f9827`` is a universally unique
identifier for this particular Segway run. This can be useful if you
want to manage all of your jobs on your clustering system with
wildcard specification. On SGE you can delete all the jobs from this
run with::

  qdel "*.ed03201cea2047399d4cbcc4b62f9827"

On LSF, use::

  bkill -J "*.ed03201cea2047399d4cbcc4b62f9827"

Jobs created in the identify (``vit``) or posterior (``jt``) task are
named similarly::

  vit34.identifydir.4f32630d53724f08b34a8fc58793307d
  jt34.identifydir.4f32630d53724f08b34a8fc58793307d

Of course, there are no instances or rounds for the identify task, so
only the sequence index is reported.

Tracks
------

Tracks are named according to their name in the Genomedata archive.
For GMTK internal use, periods are converted to underscores.
