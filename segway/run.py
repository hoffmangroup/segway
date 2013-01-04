#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: main Segway implementation
"""

__version__ = "$Revision$"

# Copyright 2008-2012 Michael M. Hoffman <mmh1@uw.edu>

import os

from collections import defaultdict, namedtuple
from copy import copy
from datetime import datetime
from distutils.spawn import find_executable
from errno import EEXIST, ENOENT
from functools import partial
from itertools import count, izip, product
from math import ceil, ldexp, log
from os import environ, extsep
import re
from shutil import copy2
from string import letters
import sys
from threading import Event, Lock, Thread
from time import sleep
from urllib import quote
from uuid import uuid1
from warnings import warn
import struct
import pdb

from genomedata import Genome
from numpy import (arange, arcsinh, array, empty, finfo, float32, intc,
                   int64, inf, square, vstack, zeros, hstack)
from optplus import str2slice_or_int
from optbuild import AddableMixin
from path import path
from pkg_resources import Requirement, working_set
from tabdelim import DictReader, ListWriter

from .bed import parse_bed4, read_native, read_native_file, read as read_bed3
from .cluster import (make_native_spec, JobTemplateFactory, RestartableJob,
                      RestartableJobDict, Session)
from .include import IncludeSaver
from .input_master import InputMasterSaver
from .observations import Observations, find_overlaps_include # XXX move find_overlaps_include to _util
from .output import IdentifySaver, PosteriorSaver
from .structure import StructureSaver
from .measure_prop import MeasurePropRunner
from .virtual_evidence import write_virtual_evidence
from ._util import (data_filename,
                    DTYPE_OBS_INT, DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                    DISTRIBUTION_ASINH_NORMAL, DISTRIBUTION_POWER_NORM, EXT_BED, EXT_FLOAT, EXT_GZ,
                    EXT_INT, EXT_PARAMS, EXT_TAB, EXT_LIST,
                    extjoin, extjoin_not_none, GB,
                    ISLAND_BASE_NA, ISLAND_LST_NA, load_coords,
                    make_default_filename,
                    make_filelistpath, make_prefix_fmt, ceildiv,
                    permissive_log,
                    MB, memoized_property, OFFSET_START, OFFSET_END,
                    OFFSET_STEP, OptionBuilder_GMTK, PassThroughDict,
                    POSTERIOR_PROG, PREFIX_LIKELIHOOD, PREFIX_PARAMS,
                    PREFIX_MP_OBJ,
                    SEG_TABLE_WIDTH, SUBDIRNAME_LOG, SUBDIRNAME_PARAMS,
                    SUPERVISION_LABEL_OFFSET,
                    SUPERVISION_UNSUPERVISED, SUPERVISION_SEMISUPERVISED,
                    USE_MFSDG, VITERBI_PROG,
                    VIRTUAL_EVIDENCE_FULL_LIST_FILENAME,
                    VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME_TMPL,
                    VIRTUAL_EVIDENCE_OBS_FILENAME_TMPL,
                    FilesGenome, FilesChromosome, FILE_TRACKS_SENTINEL)

# set once per file run
UUID = uuid1().hex

# XXX: I should really get some sort of Enum for this, I think Peter
# Norvig has one
#DISTRIBUTIONS = [DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                 #DISTRIBUTION_ASINH_NORMAL]
DISTRIBUTION_DEFAULT = DISTRIBUTION_ASINH_NORMAL

MIN_NUM_SEGS = 2
NUM_SEGS = MIN_NUM_SEGS
NUM_SUBSEGS = 1
RULER_SCALE = 10
MAX_EM_ITERS = 100
TEMPDIR_PREFIX = __package__ + "-"

ISLAND = True

# XXX: temporary code to allow easy switching
if ISLAND:
    ISLAND_BASE = 3
    ISLAND_LST = 100000
    HASH_LOAD_FACTOR = 0.98
else:
    ISLAND_BASE = ISLAND_BASE_NA
    ISLAND_LST = ISLAND_LST_NA
    HASH_LOAD_FACTOR = None

COMPONENT_CACHE = not ISLAND
DETERMINISTIC_CHILDREN_STORE = not ISLAND

assert (ISLAND or
        (ISLAND_LST == ISLAND_LST_NA and ISLAND_BASE == ISLAND_BASE_NA))

LINEAR_MEM_USAGE_MULTIPLIER = 1
MEM_USAGE_MULTIPLIER = 2

JOIN_TIMEOUT = finfo(float).max

SWAP_ENDIAN = False

## option defaults
VERBOSITY = 0
LEN_PRIOR_STRENGTH = 0
GRAPH_PRIOR_STRENGTH = 0

FINFO_FLOAT32 = finfo(float32)
MACHEP_FLOAT32 = FINFO_FLOAT32.machep
TINY_FLOAT32 = FINFO_FLOAT32.tiny

SIZEOF_FLOAT32 = float32().nbytes
SIZEOF_DTYPE_OBS_INT = DTYPE_OBS_INT().nbytes

# sizeof tmp space used per frame
SIZEOF_FRAME_TMP = (SIZEOF_FLOAT32 + SIZEOF_DTYPE_OBS_INT)

FUDGE_EP = -17 # ldexp(1, -17) = ~1e-6
assert FUDGE_EP > MACHEP_FLOAT32

FUDGE_TINY = -ldexp(TINY_FLOAT32, 6)

# This is looser criterion is supported
# by seed-level variation
LOG_LIKELIHOOD_DIFF_FRAC = 1e-3
#LOG_LIKELIHOOD_DIFF_FRAC = 1e-5
#LOG_LIKELIHOOD_DIFF_FRAC = 1e-7

NUM_SEQ_COLS = 2 # dinucleotide, presence_dinucleotide

MAX_FRAMES = 2000000 # 2 million
MEM_USAGE_BUNDLE = 100*MB # XXX: should start using this again
MEM_USAGE_PROGRESSION = "2,3,4,6,8,10,12,14,15"

TMP_USAGE_BASE = 10*MB # just a guess

POSTERIOR_CLIQUE_INDICES = dict(p=1, c=1, e=1)

## defaults
NUM_INSTANCES = 1

CPP_DIRECTIVE_FMT = "-D%s=%s"

GMTK_INDEX_PLACEHOLDER = "@D"
NAME_BUNDLE_PLACEHOLDER = "bundle"

# programs
ENV_CMD = "/usr/bin/env"

# XXX: need to differentiate this (prog) from prog.prog == progname throughout
TRIANGULATE_PROG = OptionBuilder_GMTK("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_GMTK("gmtkEMtrain")

TMP_OBS_PROGS = frozenset([VITERBI_PROG, POSTERIOR_PROG])

SPECIAL_TRACKNAMES = frozenset(["dinucleotide", "supervisionLabel"])

# extensions and suffixes
EXT_BEDGRAPH = "bedGraph"
EXT_BIN = "bin"
EXT_LIKELIHOOD = "ll"
EXT_LOG = "log"
EXT_OUT = "out"
EXT_POSTERIOR = "posterior"
EXT_SH = "sh"
EXT_TXT = "txt"
EXT_TRIFILE = "trifile"

PREFIX_ACC = "acc"
PREFIX_CMDLINE_SHORT = "run"
PREFIX_CMDLINE_LONG = "details"
PREFIX_CMDLINE_TOP = "segway"
PREFIX_TRAIN = "train"
PREFIX_POSTERIOR = "posterior%s"

PREFIX_VITERBI = "viterbi"
PREFIX_WINDOW = "window"
PREFIX_JT_INFO = "jt_info"
PREFIX_JOB_LOG = "jobs"

PREFIX_JOB_NAME_TRAIN = "emt"
PREFIX_JOB_NAME_VITERBI = "vit"
PREFIX_JOB_NAME_POSTERIOR = "jt"

SUFFIX_OUT = extsep + EXT_OUT
SUFFIX_TRIFILE = extsep + EXT_TRIFILE

BED_FILEBASENAME = extjoin(__package__, EXT_BED, EXT_GZ) # "segway.bed.gz"
BED_FILEBASEFMT = extjoin(__package__, "%d", EXT_BED, EXT_GZ) # "segway.%d.bed.gz"
BEDGRAPH_FILEBASENAME = extjoin(PREFIX_POSTERIOR, EXT_BEDGRAPH, EXT_GZ) # "posterior%s.bed.gz"
BEDGRAPH_FILEBASEFMT = extjoin(PREFIX_POSTERIOR, "%%d", EXT_BEDGRAPH, EXT_GZ) # "posterior%s.%%d.bed.gz"
FLOAT_TABFILEBASENAME = extjoin("observations", EXT_TAB)
TRAIN_FILEBASENAME = extjoin(PREFIX_TRAIN, EXT_TAB)

SUBDIRNAME_ACC = "accumulators"
SUBDIRNAME_AUX = "auxiliary"
SUBDIRNAME_LIKELIHOOD = "likelihood"
SUBDIRNAME_OBS = "observations"
SUBDIRNAME_POSTERIOR = "posterior"
SUBDIRNAME_VITERBI = "viterbi"
SUBDIRNAME_MEASURE_PROP = "measure_prop"

SUBDIRNAMES_EITHER = [SUBDIRNAME_AUX]
SUBDIRNAMES_TRAIN = [SUBDIRNAME_ACC, SUBDIRNAME_LIKELIHOOD,
                     SUBDIRNAME_PARAMS]

JOB_LOG_FIELDNAMES = ["jobid", "jobname", "prog", "num_segs",
                      "num_frames", "maxvmem", "cpu", "exit_status"]
# XXX: should add num_subsegs as well, but it's complicated to pass
# that data into RestartableJobDict.wait()

TRAIN_FIELDNAMES = ["name", "value"]

TRAIN_OPTION_TYPES = \
    dict(input_master_filename=str, structure_filename=str,
         params_filename=str, dont_train_filename=str, seg_table_filename=str,
         distribution=str, len_seg_strength=float, graph_seg_strength=float,
         segtransition_weight_scale=float, ruler_scale=int, resolution=int,
         num_segs=int, num_subsegs=int, track_specs=[str],
         reverse_worlds=[int])

# templates and formats
RES_OUTPUT_MASTER = "output.master"
RES_DONT_TRAIN = "dont_train.list"
RES_SEG_TABLE = "seg_table.tab"

TRAIN_ATTRNAMES = ["input_master_filename", "params_filename",
                   "log_likelihood_filename"]
LEN_TRAIN_ATTRNAMES = len(TRAIN_ATTRNAMES)

COMMENT_POSTERIOR_TRIANGULATION = \
    "%% triangulation modified for posterior decoding by %s" % __package__

RESOLUTION = 1

SEGTRANSITION_WEIGHT_SCALE = 1.0

DIRPATH_WORK_DIR_HELP = path("WORKDIR")

# 62 so that it's not in sync with the 10 second job wait sleep time
#THREAD_START_SLEEP_TIME = 62 # XXX: this should become an option
THREAD_START_SLEEP_TIME = 2 # XXX: this should become an option # XXX

# -gpr option for GMTK when reversing
REVERSE_GPR = "^0:-1:0"

Results = namedtuple("Results", ["instance_index", "log_likelihood", "mp_terms", "num_segs",
                                 "input_master_filename", "params_filename",
                                 "log_likelihood_filename"])
OFFSET_FILENAMES = 4 # where the filenames begin in Results

# Used to compare Results objects in order to find the winning instance
def results_objective_cmp(res1, res2):
    return cmp(objective_value(res1.log_likelihood, res1.mp_terms),
               objective_value(res2.log_likelihood, res2.mp_terms))


## functions
def quote_trackname(text):
    # legal characters are ident in GMTK_FileTokenizer.ll:
    # {alpha})({alpha}|{dig}|\_|\-)* (alpha is [A-za-z], dig is [0-9])
    res = text.replace("_", "_5F")
    res = res.replace(".", "_2E")

    # quote eliminates everything that doesn't match except for "_.-",
    # replaces with % escapes
    res = quote(res, safe="") # safe="" => quote "/" too
    res = res.replace("%", "_")

    # add stub to deal with non-alphabetic first characters
    if res[0] not in letters:
        # __ should never appear in strings quoted as before
        res = "x__" + res

    return res

def quote_spaced_str(item):
    """
    add quotes around text if it has spaces in it
    """
    text = str(item)

    if " " in text:
        return '"%s"' % text
    else:
        return text

class NoAdvance(str):
    """
    cause rewrite_strip_comments() to not consume an extra line
    """

class NewLine(NoAdvance):
    """
    add a line rather than replacing existing
    """
    # doesn't actually have any code. used solely for class identification

def rewrite_strip_comments(infile, outfile):
    """
    strips comments, and trailing whitespace from lines
    """
    for line in infile:
        inline = line.rstrip()

        if not inline or inline.startswith("%"):
            outline = inline
        else:
            outline = (yield inline)

            if isinstance(outline, NewLine):
                print >>outfile, inline
            elif outline is None:
                outline = inline

        print >>outfile, outline

        while isinstance(outline, NoAdvance):
            outline = (yield)

            if outline is not None:
                print >>outfile, outline

def consume_until(iterable, text):
    for line in iterable:
        if line.startswith(text):
            break

def slice2range(s):
    if isinstance(s, int):
        return [s]

    start = s.start
    stop = s.stop
    step = s.step

    # need to know the length of the sequence to work with stop == None
    assert stop is not None

    if start is None:
        start = 0

    if step is None:
        step = 1

    return xrange(start, stop, step)

def is_training_progressing(last_ll, curr_ll,
                            min_ll_diff_frac=LOG_LIKELIHOOD_DIFF_FRAC):
    # using x !< y instead of x >= y to give the right default answer
    # in the case of NaNs

    return not abs((curr_ll - last_ll)/last_ll) < min_ll_diff_frac

def objective_value(log_likelihood, mp_terms):
    if mp_terms:
        return log_likelihood - (mp_terms[1] - mp_terms[2])
    else:
        return log_likelihood


def set_cwd_job_tmpl(job_tmpl):
    job_tmpl.workingDirectory = path.getcwd()

def rewrite_cliques(rewriter, frame):
    """
    returns the index of the added clique
    """
    # method
    rewriter.next()

    # number of cliques
    orig_num_cliques = int(rewriter.next())
    rewriter.send(NoAdvance(orig_num_cliques + 1))

    # original cliques
    for clique_index in xrange(orig_num_cliques):
        rewriter.next()

    # new clique
    rewriter.send(NewLine("%d 1 seg %d" % (orig_num_cliques, frame)))

    # XXX: add subseg as a clique to report it in posterior

    return orig_num_cliques

def make_mem_req(mem_usage):
    # double usage at this point
    mem_usage_gibibytes = ceil(mem_usage / GB)

    return "%dG" % mem_usage_gibibytes

class Mixin_Lockable(AddableMixin):
    def __init__(self, *args, **kwargs):
        self.lock = Lock()
        return AddableMixin.__init__(self, *args, **kwargs)

LockableDefaultDict = Mixin_Lockable + defaultdict

class TrainThread(Thread):
    def __init__(self, runner, session, instance_index, num_segs):
        # keeps it from rewriting variables that will be used
        # later or in a different thread
        self.runner = copy(runner)

        self.session = session
        self.num_segs = num_segs
        self.instance_index = instance_index

        Thread.__init__(self)

    def run(self):
        self.runner.session = self.session
        self.runner.num_segs = self.num_segs
        self.runner.instance_index = self.instance_index
        self.result = self.runner.run_train_instance()

def maybe_quote_arg(text):
    """
    return quoted argument, adding backslash quotes

    XXX: would be nice if this were smarter about what needs to be
    quoted, only doing this when special characters or whitespace are
    within the arg

    XXX: Enclosing characters in double quotes (`"') preserves the literal value
of all characters within the quotes, with the exception of `$', ``',
`\', and, when history expansion is enabled, `!'.  The characters `$'
and ``' retain their special meaning within double quotes (*note Shell
Expansions::).  The backslash retains its special meaning only when
followed by one of the following characters: `$', ``', `"', `\', or
`newline'.  Within double quotes, backslashes that are followed by one
of these characters are removed.  Backslashes preceding characters
without a special meaning are left unmodified.  A double quote may be
quoted within double quotes by preceding it with a backslash.  If
enabled, history expansion will be performed unless an `!' appearing in
double quotes is escaped using a backslash.  The backslash preceding
the `!' is not removed.
    """
    return '"%s"' % text.replace('"', r'\"')

def cmdline2text(cmdline=sys.argv):
    return " ".join(maybe_quote_arg(arg) for arg in cmdline)

def _log_cmdline(logfile, cmdline):
    print >>logfile, " ".join(map(quote_spaced_str, cmdline))

def check_overlapping_supervision_labels(start, end, chrom, coords):
    for coord_start, coord_end in coords[chrom]:
        if not (coord_start >= end or coord_end <= start):
            raise ValueError("supervision label %s(%s, %s) overlaps"
                             "supervision label %s(%s, %s)" %
                             (chrom, coord_start, coord_end,
                              chrom, start, end))

re_num_cliques = re.compile(r"^Number of cliques = (\d+)$")
re_clique_info = re.compile(r"^Clique information: .*, (\d+) unsigned words ")
class Runner(object):
    """
    Purpose:

    1. hold configuration--interface between UI and other code
    2. create necessary files (through Saver objects)
       TODO: move all saving to Saver objects
    3. execute GMTK
    4. monitor GMTK output
    5. convert output to bioinformatics formats
       TODO: move this to some other kind of object
    """
    def __init__(self, **kwargs):
        """
        usually not called directly, instead Runner.fromoptions() is called
        (which calls Runner.__init__())
        """
        self.uuid = UUID

        # filenames
        self.bigbed_filename = None
        self.gmtk_include_filename = None
        self.input_master_filename = None
        self.structure_filename = None
        self.triangulation_filename = None
        self.job_log_filename = None
        self.seg_table_filename = None
        self.supervision_filename = None

        self.params_filename = None # actual params filename for this instance
        self.params_filenames = [] # list of possible params filenames
        self.recover_dirname = None
        self.work_dirname = None
        self.log_likelihood_filename = None
        self.log_likelihood_tab_filename = None

        self.obs_dirname = None

        self.include_coords_filename = None
        self.exclude_coords_filename = None
        self.enforce_coords = False

        self.posterior_clique_indices = POSTERIOR_CLIQUE_INDICES.copy()

        self.triangulation_filename_is_new = None

        self.supervision_coords = None
        self.supervision_labels = None

        # XXXmax
        self.virtual_evidence_filename = None
        self.measure_prop_graph_filepath = False
        self.mp_runner = None
        self.measure_prop_ve_dirpath = None # set by MeasurePropRunner
        self.mu = None
        self.nu = None
        self.mp_weight = None
        self.measure_prop_objective_tab_filename = None
        self.measure_prop_reuse_evidence = False

        self.card_supervision_label = -1

        self.include_tracknames = []
        self.tied_tracknames = {} # dict of head trackname -> tied tracknames
        self.head_trackname_list = [] # ordered list of tied_trackname.keys()

        # default is 0
        self.global_mem_usage = LockableDefaultDict(int)

        # data
        # a "window" is what GMTK calls a segment
        self.windows = None
        self.mins = None
        self.maxs = None
        self.tracknames = None # encoded/quoted version
        self.track_specs = []

        # variables
        self.num_segs = NUM_SEGS
        self.num_subsegs = NUM_SUBSEGS
        self.num_instances = NUM_INSTANCES
        self.len_seg_strength = LEN_PRIOR_STRENGTH
        self.graph_seg_strength = GRAPH_PRIOR_STRENGTH
        self.distribution = DISTRIBUTION_DEFAULT
        self.max_em_iters = MAX_EM_ITERS
        self.max_frames = MAX_FRAMES
        self.segtransition_weight_scale = SEGTRANSITION_WEIGHT_SCALE
        self.ruler_scale = RULER_SCALE
        self.resolution = RESOLUTION
        self.reverse_worlds = [] # XXXopt: this should be a set

        # flags
        self.clobber = False
        self.train = False # EM train # this should become an int for num_starts
        self.posterior = False
        self.identify = False # viterbi
        self.dry_run = False
        self.verbosity = VERBOSITY
        self.use_dinucleotide = None

        self.__dict__.update(kwargs)

    def set_tasks(self, text):
        tasks = text.split("+")
        if "train" in tasks and len(tasks) > 1:
            raise ValueError("train task must be run separately")

        for task in tasks:
            if task == "train":
                self.train = True
            elif task == "identify":
                self.identify = True
            elif task == "posterior":
                self.posterior = True
            else:
                raise ValueError("unrecognized task: %s" % task)

    def set_option(self, name, value):
        if value or value == 0 or value is False or value == []:
            setattr(self, name, value)

    options_to_attrs = [("recover", "recover_dirname"),
                        ("observations", "obs_dirname"),
                        ("bed", "bed_filename"),
                        ("semisupervised", "supervision_filename"),
                        ("measure_prop","measure_prop_graph_filepath"),
                        ("measure_prop_mu","mu"),
                        ("measure_prop_nu","nu"),
                        ("measure_prop_weight","mp_weight"),
                        ("measure_prop_num_iters",),
                        ("measure_prop_am_num_iters",),
                        ("measure_prop_reuse_evidence",),
                        ("virtual_evidence", "virtual_evidence_filename"),
                        ("bigBed", "bigbed_filename"),
                        ("include_coords", "include_coords_filename"),
                        ("exclude_coords", "exclude_coords_filename"),
                        ("enforce_coords",),
                        ("num_instances",),
                        ("verbosity",),
                        ("split_sequences", "max_frames"),
                        ("clobber",),
                        ("file_tracks",),
                        ("dry_run",),
                        ("input_master", "input_master_filename"),
                        ("structure", "structure_filename"),
                        ("dont_train", "dont_train_filename"),
                        ("seg_table", "seg_table_filename"),
                        ("distribution",),
                        ("len_prior_strength", "len_seg_strength"),
                        ("graph_prior_strength", "graph_seg_strength"),
                        ("segtransition_weight_scale",),
                        ("ruler_scale",),
                        ("resolution",),
                        ("num_labels", "num_segs"),
                        ("num_sublabels", "num_subsegs"),
                        ("max_train_rounds", "max_em_iters"),
                        ("reverse_world", "reverse_worlds"),
                        ("track", "track_specs")]

    @classmethod
    def fromargs(cls, args):
        res = cls()

        task_str = args[0]
        genomedataname = args[1]
        traindirname = args[2]

        res.set_tasks(task_str)
        res.genomedataname = genomedataname

        if res.train:
            res.work_dirname = traindirname
            assert len(args) == 3
            return res

        # identify or posterior
        res.work_dirname = args[3]

        try:
            res.load_train_options(traindirname)
        except IOError, err:
            # train.tab use is optional
            if err.errno != ENOENT:
                raise

        return res

    @classmethod
    def fromoptions(cls, args, options):
        """
        the usual way a Runner is created
        """
        res = cls.fromargs(args)

        # bulk copy options that need no further processing
        for option_to_attr in cls.options_to_attrs:
            try:
                src, dst = option_to_attr
            except ValueError:
                src, = option_to_attr
                dst = src

            res.set_option(dst, getattr(options, src))

        # multiple lists to one
        res.user_native_spec = sum([opt.split(" ")
                                    for opt in options.cluster_opt], [])

        mem_usage_list = map(float, options.mem_usage.split(","))

        # XXX: should do a ceil first?
        # use int64 in case run.py is run on a 32-bit machine to control
        # 64-bit compute nodes
        res.mem_usage_progression = (array(mem_usage_list) * GB).astype(int64)

        # don't change from None if this is false
        params_filenames = options.trainable_params
        if params_filenames:
            res.params_filenames = params_filenames

        include_tracknames = []

        # dict. key: str: head trackname; value: list(str: tracknames)
        tied_tracknames = defaultdict(list)

        # dict. key: str: trackname; value: head trackname
        # inverse of tied_tracknames
        head_tracknames = {}

        # ordered list of the head tracknames
        head_trackname_list = []

        # temporary list to avoid duplicates
        used_tracknames = set()

        for track_spec in res.track_specs:
            # local to each value of track_spec
            current_tracknames = track_spec.split(",")
            current_tracknames_set = frozenset(current_tracknames)

            assert len(current_tracknames_set) == len(current_tracknames)

            if not used_tracknames.isdisjoint(current_tracknames_set):
                raise ValueError("can't tie one track in multiple groups")

            include_tracknames.extend(current_tracknames)
            used_tracknames |= current_tracknames_set

            head_trackname = current_tracknames[0]
            head_trackname_list.append(head_trackname)
            for trackname in current_tracknames:
                tied_tracknames[head_trackname].append(trackname)
                head_tracknames[trackname] = head_trackname

        if include_tracknames:
            # non-allowed special trackname
            # XXX: doesn't deal with the case where it is a default
            # trackname in input file
            assert "supervisionLabel" not in include_tracknames
            res.include_tracknames = include_tracknames

        res.tied_tracknames = tied_tracknames

        if head_tracknames:
            res.head_tracknames = head_tracknames
            res.head_trackname_list = head_trackname_list

        return res

    @memoized_property
    def triangulation_dirpath(self):
        res = self.work_dirpath / "triangulation"
        self.make_dir(res)

        return res

    @memoized_property
    def jt_info_filename(self):
        return self.make_filename(PREFIX_JT_INFO, EXT_TXT,
                                  subdirname=SUBDIRNAME_LOG)

    @memoized_property
    def posterior_jt_info_filename(self):
        return self.make_filename(PREFIX_JT_INFO, "posterior", EXT_TXT,
                                  subdirname=SUBDIRNAME_LOG)

    @memoized_property
    def winner_filename(self):
        return self.make_filename("winner")

    @memoized_property
    def virtual_evidence_dirpath(self):
        res = self.make_filename("virtual_evidence", subdirname=SUBDIRNAME_OBS)
        self.make_dir(res)
        return res

    @memoized_property
    def virtual_evidence_ve_full_list_filename(self):
        return (self.virtual_evidence_dirpath / VIRTUAL_EVIDENCE_FULL_LIST_FILENAME)

    @memoized_property
    def virtual_evidence_ve_window_list_filenames(self):
        return [((self.virtual_evidence_dirpath / VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME_TMPL)
                 % window_index) for window_index in range(len(self.windows))]

    @memoized_property
    def virtual_evidence_ve_obs_filenames(self):
        return [((self.virtual_evidence_dirpath / VIRTUAL_EVIDENCE_OBS_FILENAME_TMPL)
                 % window_index) for window_index in range(len(self.windows))]

    @memoized_property
    def uniform_ve_dirname(self):
        res = self.obs_dirpath / "uniform_ve"
        if not res.isdir():
            self.make_dir(res)
        return res

    def make_measure_prop_ve_full_list_filename(self, instance_index, round_index):
        return (self.measure_prop_ve_dirpath
                / VIRTUAL_EVIDENCE_FULL_LIST_FILENAME)

    def make_measure_prop_ve_window_list_filenames(self, instance_index, round_index):
        return [((self.measure_prop_ve_dirpath
                 / VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME_TMPL)
                 % window_index) for window_index in range(len(self.windows))]

    def make_measure_prop_ve_obs_filenames(self):
        return [((self.measure_prop_ve_dirpath
                 / VIRTUAL_EVIDENCE_OBS_FILENAME_TMPL)
                 % window_index) for window_index in range(len(self.windows))]

    def make_mp_posterior_workdir(self, instance_index, round_index):
        res = self.posterior_dirname / ("measure_prop.%s.%s" % (instance_index, round_index))
        if not res.isdir():
            self.make_dir(res)
        return res

    @memoized_property
    def work_dirpath(self):
        return path(self.work_dirname)

    @memoized_property
    def recover_dirpath(self):
        recover_dirname = self.recover_dirname
        if recover_dirname:
            return path(recover_dirname)

        # recover_dirname is None or ""
        return None

    @memoized_property
    def include_coords(self):
        return read_native_file(self.include_coords_filename)

    @memoized_property
    def exclude_coords(self):
        return read_native_file(self.exclude_coords_filename)

    @memoized_property
    def seg_table(self):
        filename = self.seg_table_filename

        if not filename:
            filename = data_filename("seg_table.tab")

        # always the last element of the range
        num_segs = slice2range(self.num_segs)[-1]

        res = zeros((num_segs, SEG_TABLE_WIDTH), dtype=int)
        ruler_scale = self.ruler_scale
        res[:, OFFSET_STEP] = ruler_scale

        with open(filename) as infile:
            reader = DictReader(infile)

            # overwriting is allowed
            for row in reader:
                # XXX: factor out
                # get row_indexes
                label = row["label"]
                label_slice = str2slice_or_int(label)

                if isinstance(label_slice, slice) and label_slice.stop is None:
                    label_slice = slice(label_slice.start, num_segs,
                                        label_slice.step)

                row_indexes = slice2range(label_slice)

                # get slice
                len_slice = str2slice_or_int(row["len"])

                # XXX: eventually, should read ruler scale from file
                # instead of using as a command-line option
                try:
                    assert len_slice.step == ruler_scale
                except:
                    pdb.set_trace() # XXX

                len_tuple = (len_slice.start, len_slice.stop, len_slice.step)
                len_row = zeros((SEG_TABLE_WIDTH))

                for item_index, item in enumerate(len_tuple):
                    if item is not None:
                        len_row[item_index] = item

                res[row_indexes] = len_row

        return res

    @memoized_property
    def obs_dirpath(self):
        obs_dirname = self.obs_dirname

        if obs_dirname:
            res = path(obs_dirname)
        else:
            res = self.work_dirpath / SUBDIRNAME_OBS
            self.obs_dirname = res

        try:
            self.make_dir(res)
        except OSError, err:
            if not (err.errno == EEXIST and res.isdir()):
                raise

        return res

    @memoized_property
    def float_filelistpath(self):
        return self.make_obs_filelistpath(EXT_FLOAT)

    @memoized_property
    def int_filelistpath(self):
        return self.make_obs_filelistpath(EXT_INT)

    @memoized_property
    def float_tabfilepath(self):
        return self.obs_dirpath / FLOAT_TABFILEBASENAME

    @memoized_property
    def gmtk_include_filename_relative(self):
        return self.gmtk_include_filename

        # XXX: disable until you figure out a good way of dealing with
        # includes from params/input.master as well

        # dirpath_trailing_slash = self.work_dirpath + "/"
        # include_filename_relative = \
        #     include_filename.partition(dirpath_trailing_slash)[2]
        # assert include_filename_relative

        #self.gmtk_include_filename_relative = include_filename_relative

    @memoized_property
    def _means_untransformed(self):
        return self.sums / self.num_datapoints

    @memoized_property
    def means(self):
        return self.transform(self._means_untransformed)

    @memoized_property
    def vars(self):
        # this is an unstable way of calculating the variance,
        # but it should be good enough
        # Numerical Recipes in C, Eqn 14.1.7
        # XXX: best would be to switch to the pairwise parallel method
        # (see Wikipedia)

        sums_squares_normalized = self.sums_squares / self.num_datapoints
        return self.transform(sums_squares_normalized - square(self._means_untransformed))

    @memoized_property
    def dont_train_filename(self):
        return self.save_resource(RES_DONT_TRAIN, SUBDIRNAME_AUX)

    @memoized_property
    def output_master_filename(self):
        return self.save_resource(RES_OUTPUT_MASTER, SUBDIRNAME_PARAMS)

    def make_viterbi_filenames(self, dirpath):
        """
        make viterbi filenames for a particular dirpath
        """
        viterbi_dirpath = dirpath / SUBDIRNAME_VITERBI
        num_windows = self.num_windows

        viterbi_filename_fmt = (PREFIX_VITERBI + make_prefix_fmt(num_windows)
                                + EXT_BED + "." + EXT_GZ)
        return [viterbi_dirpath / viterbi_filename_fmt % index
                for index in xrange(num_windows)]

    @memoized_property
    def viterbi_filenames(self):
        self.make_subdir(SUBDIRNAME_VITERBI)
        return self.make_viterbi_filenames(self.work_dirpath)

    @memoized_property
    def recover_viterbi_filenames(self):
        recover_dirpath = self.recover_dirpath
        if recover_dirpath:
            return self.make_viterbi_filenames(recover_dirpath)
        else:
            return None

    @memoized_property
    def posterior_dirname(self):
        self.make_subdir(SUBDIRNAME_POSTERIOR)
        return path(self.work_dirpath) / SUBDIRNAME_POSTERIOR

    @memoized_property
    def posterior_filenames(self):
        self.posterior_dirname
        return map(self.make_posterior_filename, xrange(self.num_windows))

    @memoized_property
    def recover_posterior_filenames(self):
        raise NotImplementedError # XXX

    @memoized_property
    def params_dirpath(self):
        return self.work_dirpath / SUBDIRNAME_PARAMS

    @memoized_property
    def recover_params_dirpath(self):
        recover_dirpath = self.recover_dirpath
        if recover_dirpath:
            return recover_dirpath / SUBDIRNAME_PARAMS

    @memoized_property
    def window_lens(self):
        return [len(window) for window in self.windows]

    @memoized_property
    def posterior_triangulation_filename(self):
        infilename = self.triangulation_filename

        # either strip ".trifile" off end, or just use the whole filename
        infilename_stem = (infilename.rpartition(SUFFIX_TRIFILE)[0]
                           or infilename)

        res = extjoin(infilename_stem, EXT_POSTERIOR, EXT_TRIFILE)

        clique_indices = self.posterior_clique_indices

        # XXX: this is a fairly hacky way of doing it and will not
        # work if the triangulation file changes from what GMTK
        # generates. probably need to key on tokens rather than lines
        with open(infilename) as infile:
            with open(res, "w") as outfile:
                print >>outfile, COMMENT_POSTERIOR_TRIANGULATION
                rewriter = rewrite_strip_comments(infile, outfile)

                consume_until(rewriter, "@@@!!!TRIFILE_END_OF_ID_STRING!!!@@@")
                consume_until(rewriter, "CE_PARTITION")

                components_indexed = enumerate(POSTERIOR_CLIQUE_INDICES)
                for component_index, component in components_indexed:
                    clique_index = rewrite_cliques(rewriter, component_index)
                    clique_indices[component] = clique_index

                for line in rewriter:
                    pass

        return res

    @memoized_property
    def output_dirpath(self):
        return self.make_output_dirpath("o", self.instance_index)

    @memoized_property
    def error_dirpath(self):
        return self.make_output_dirpath("e", self.instance_index)

    @memoized_property
    def use_dinucleotide(self):
        return "dinucleotide" in self.include_tracknames

    @memoized_property
    def num_int_cols(self):
        if not USE_MFSDG or self.resolution > 1:
            res = self.num_tracks
        else:
            res = 0

        if self.use_dinucleotide:
            res += NUM_SEQ_COLS
        if self.supervision_type != SUPERVISION_UNSUPERVISED:
            res += 1

        return res

    @memoized_property
    def bed_filename(self):
        if self.num_worlds == 1:
            basename = BED_FILEBASENAME
        else:
            basename = BED_FILEBASEFMT

        return self.work_dirpath / basename

    @memoized_property
    def bedgraph_filename(self):
        if self.num_worlds == 1:
            basename = BEDGRAPH_FILEBASENAME
        else:
            basename = BEDGRAPH_FILEBASEFMT

        return self.work_dirpath / basename

    @memoized_property
    def train_prog(self):
        return self.prog_factory(EM_TRAIN_PROG)

    @memoized_property
    def seg_countdowns_initial(self):
        table = self.seg_table

        starts = table[:, OFFSET_START]
        ends = table[:, OFFSET_END]
        steps = table[:, OFFSET_STEP]

        # XXX: need to assert that ends are either 0 or are always
        # greater than starts

        # starts and ends must all be divisible by steps
        assert not (starts % steps).any()
        assert not (ends % steps).any()

        # // = floor division
        seg_countdowns_start = starts // steps

        # need minus one to guarantee maximum
        seg_countdowns_end = (ends // steps) - 1

        seg_countdowns_both = vstack([seg_countdowns_start,
                                      seg_countdowns_end])

        return seg_countdowns_both.max(axis=0)

    @memoized_property
    def card_seg_countdown(self):
        return self.seg_countdowns_initial.max() + 1

    @memoized_property
    def num_tracks(self):
        return len(self.head_trackname_list)

    @memoized_property
    def num_windows(self):
        return len(self.windows)

    @memoized_property
    def num_bases(self):
        return sum(self.window_lens)

    @memoized_property
    def head_tracknames(self):
        return PassThroughDict()

    @memoized_property
    def head_trackname_list(self):
        return self.tracknames

    @memoized_property
    def supervision_type(self):
        if self.supervision_filename:
            return SUPERVISION_SEMISUPERVISED
        else:
            return SUPERVISION_UNSUPERVISED

    @memoized_property
    def virtual_evidence(self):
        if not self.virtual_evidence_filename is None:
            return True
        else:
            return False

    @memoized_property
    def world_tracknames(self):
        # XXX: add support for some heads having only one trackname
        # that is repeated

        return list(zip(*self.tied_tracknames.values()))

    @memoized_property
    def world_track_indexes(self):
        """
        all track indexes, not just the heads
        """
        assert (not self.tied_tracknames
                or len(self.tied_tracknames) == len(self.tied_track_indexes_list))

        res = array(zip(*self.tied_track_indexes_list))
        assert len(res) == self.num_worlds

        if __debug__:
            if self.num_worlds == 1:
                assert (res == self.track_indexes).all()

        return res

    @memoized_property
    def num_worlds(self):
        return len(self.world_tracknames)

    @memoized_property
    def instance_make_new_params(self):
        """
        should I make new parameters in each instance?
        """
        return self.num_instances > 1 or isinstance(self.num_segs, slice)

    @memoized_property
    def num_segs_range(self):
        return slice2range(self.num_segs)

    def transform(self, num):
        # XXX this is duplicative of the code in observations, and wrong
        if self.distribution == DISTRIBUTION_ASINH_NORMAL:
            return arcsinh(num)
        elif self.distribution[:len(DISTRIBUTION_POWER_NORM)] == DISTRIBUTION_POWER_NORM:
            power = float(self.distribution[len(DISTRIBUTION_POWER_NORM):])
            return num ** power
        else:
            return num

    def make_cpp_options(self, task, instance_index="identify", round_index="identify",
                         window_index="all", input_params_filename=None,
                         output_params_filename=None):
        directives = {}

        if input_params_filename:
            directives["INPUT_PARAMS_FILENAME"] = input_params_filename

        if output_params_filename:
            directives["OUTPUT_PARAMS_FILENAME"] = output_params_filename

        directives["CARD_SEG"] = self.num_segs
        directives["CARD_SUBSEG"] = self.num_subsegs
        directives["CARD_FRAMEINDEX"] = self.max_frames
        directives["SEGTRANSITION_WEIGHT_SCALE"] = \
            self.segtransition_weight_scale

        if task == "train":
            if self.measure_prop_graph_filepath:
                directives["MEAUSURE_PROP_VE_LIST_FILENAME"] = self.make_measure_prop_ve_full_list_filename(instance_index, round_index)
            if self.virtual_evidence:
                directives["VIRTUAL_EVIDENCE_VE_LIST_FILENAME"] = self.virtual_evidence_ve_full_list_filename
        if task == "identify":
            if self.measure_prop_graph_filepath:
                directives["MEAUSURE_PROP_VE_LIST_FILENAME"] = self.make_measure_prop_ve_window_list_filenames(instance_index, round_index)[window_index]
            if self.virtual_evidence:
                directives["VIRTUAL_EVIDENCE_VE_LIST_FILENAME"] = self.virtual_evidence_ve_window_list_filenames[window_index]


        res = " ".join(CPP_DIRECTIVE_FMT % item
                       for item in directives.iteritems())

        if res:
            return res

        # default: return None

    def load_log_likelihood(self):
        with open(self.log_likelihood_filename) as infile:
            log_likelihood = float(infile.read().strip())

        with open(self.log_likelihood_tab_filename, "a") as logfile:
            print >>logfile, str(log_likelihood)

        return log_likelihood

    def load_measure_prop_objective(self, round_index):
        self.last_mp_obj_filename = self.mp_runner.make_mp_obj_filename(self.instance_index, round_index)
        with open(self.last_mp_obj_filename) as obj_file:
            obj_data = dict(map(lambda line: line.split(), obj_file.readlines()))
        objective = self.mp_weight * float(obj_data["total"])
        term1 = self.mp_weight * float(obj_data["term1"])
        term2 = self.mp_weight * float(obj_data["term2"])
        term3 = self.mp_weight * float(obj_data["term3"])

        with open(self.measure_prop_objective_tab_filename, "a") as logfile:
            print >>logfile, "\t".join(map(str, [objective, term1, term2, term3]))

        return [term1, term2, term3]

    def make_filename(self, *exts, **kwargs):
        """
        makes a filename by joining together exts

        kwargs:
        dirname: top level directory (default self.work_dirname)
        subdirname: next level directory
        """
        filebasename = extjoin_not_none(*exts)

        # add subdirname if it exists
        return path(kwargs.get("dirname", self.work_dirname)) \
            / kwargs.get("subdirname", "") \
            / filebasename

    def set_tracknames(self, genome):
        # XXX: this function could use a refactor
        # there is a lot of stuff here that might not be used anywhere
        # and variable names are confusing
        tracknames = genome.tracknames_continuous

        # supplied by user: includes special tracks (like dinucleotide)
        include_tracknames = self.include_tracknames
        unquoted_tracknames = include_tracknames
        ordinary_tracknames = frozenset(trackname
                                        for trackname in include_tracknames
                                        if trackname not in SPECIAL_TRACKNAMES)
        if ordinary_tracknames:
            indexed_tracknames = [(trackname, index)
                                  for index, trackname in enumerate(tracknames)
                                  if trackname in ordinary_tracknames]

            # redefine tracknames:
            # tracknames, track_indexes = zip(*indexed_tracknames) won't return
            # ([], []) like we want
            tracknames = [indexed_trackname[0] for indexed_trackname in indexed_tracknames]
            track_indexes = [indexed_trackname[1] for indexed_trackname in indexed_tracknames]

            # check that there aren't any missing tracks
            if len(tracknames) != len(ordinary_tracknames):
                missing_tracknames = ordinary_tracknames.difference(tracknames)
                missing_tracknames_text = ", ".join(missing_tracknames)
                msg = "could not find tracknames: %s" % missing_tracknames_text
                raise ValueError(msg)

            track_indexes = array(track_indexes)
            tied_tracknames = self.tied_tracknames
            head_tracknames = self.head_tracknames
            head_trackname_list = self.head_trackname_list

            # a dict whose values are initialized in order of access
            tied_track_index_map = defaultdict(count().next)
            tied_track_indexes_list = [[] for _ in xrange(len(tied_tracknames))]

            for trackname, index in indexed_tracknames:
                head_trackname = head_tracknames[trackname]
                tied_track_indexes_list_index = tied_track_index_map[head_trackname]
                tied_track_indexes_list[tied_track_indexes_list_index].append(index)

        elif include_tracknames:
            ## no ordinary_tracknames => there are special tracknames only
            tracknames = []
            head_tracknames = {}
            head_trackname_list = []
            track_indexes = array([], intc)
            tied_track_indexes_list = []
            self.float_filelistpath = None # no float data

        else:
            # default: use all tracks in archive
            track_indexes = arange(len(tracknames))
            head_tracknames = dict(zip(tracknames, tracknames))
            head_trackname_list = tracknames

            assert not self.tied_tracknames
            self.tied_tracknames = dict((trackname, [trackname])
                                        for trackname in tracknames)

            tied_track_indexes_list = [[track_index]
                                       for track_index in track_indexes]
            unquoted_tracknames = tracknames

        # replace illegal characters in tracknames and head_tracknames only,
        # not unquoted_tracknames
        tracknames = map(quote_trackname, tracknames)
        head_tracknames = dict((quote_trackname(key), quote_trackname(value))
                               for key, value in head_tracknames.iteritems())
        head_trackname_list = map(quote_trackname, head_trackname_list)

        # assert: none of the quoted tracknames are the same
        assert len(tracknames) == len(frozenset(tracknames))

        self.tracknames = tracknames
        self.head_tracknames = head_tracknames
        self.head_trackname_list = head_trackname_list
        self.unquoted_tracknames = unquoted_tracknames
        self.track_indexes = track_indexes
        self.tied_track_indexes_list = tied_track_indexes_list

    def get_last_params_filename(self, params_filename):
        if params_filename is not None and path(params_filename).exists():
            return params_filename

        # otherwise, None is returned by default. if it doesn't exist,
        # then it's actually a new filename, but the only time this
        # will be used is when new is not set. And this will only
        # happen from the master thread.

    def make_params_filename(self, instance_index=None, dirname=None):
        if dirname is None:
            dirname = self.work_dirname

        return self.make_filename(PREFIX_PARAMS, instance_index, EXT_PARAMS,
                                  dirname=dirname,
                                  subdirname=SUBDIRNAME_PARAMS)

    def get_params_filename(self, instance_index=None, new=False):
        # this is an unexpected corner case for now
        assert not (instance_index is None and new)

        params_filenames = self.params_filenames
        num_params_filenames = len(params_filenames)

        if instance_index is None and num_params_filenames == 1:
            # special case if there is only one param filename set
            # otherwise generate "params.params" anew
            params_filename = params_filenames[0]
        elif instance_index is not None and num_params_filenames > instance_index:
            params_filename = params_filenames[instance_index]
        else:
            params_filename = None

        last_params_filename = self.get_last_params_filename(params_filename)

        # make new filenames when new is set, or params_filename is
        # not set, or the file already exists and we are training
        if (new or not params_filename
            or (self.train and path(params_filename).exists())):
            params_filename = self.make_params_filename(instance_index)

        return params_filename, last_params_filename

    def set_params_filename(self, instance_index=None, new=False):
        """
        None means the final params file, not for any particular thread
        """
        # if this is not run and params_filename is
        # unspecified, then it won't be passed to gmtkViterbiNew

        self.params_filename, self.last_params_filename = \
            self.get_params_filename(instance_index, new)

    def make_log_likelihood_tab_filename(self, instance_index, dirname):
        return self.make_filename(PREFIX_LIKELIHOOD, instance_index, EXT_TAB,
                                  dirname=dirname,
                                  subdirname=SUBDIRNAME_LOG)

    def make_measure_prop_objective_tab_filename(self, instance_index, dirname):
        return self.make_filename(PREFIX_MP_OBJ, instance_index, EXT_TAB,
                                  dirname=dirname,
                                  subdirname=SUBDIRNAME_LOG)

    def set_log_likelihood_filenames(self, instance_index=None, new=False):
        if new or not self.log_likelihood_filename:
            log_likelihood_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, instance_index,
                                   EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.log_likelihood_filename = log_likelihood_filename

            self.log_likelihood_tab_filename = \
                self.make_log_likelihood_tab_filename(instance_index,
                                                      self.work_dirname)

            if self.measure_prop_graph_filepath:
                self.measure_prop_objective_tab_filename = \
                    self.make_measure_prop_objective_tab_filename(instance_index,
                                                                  self.work_dirname)

    def make_output_dirpath(self, dirname, instance_index):
        res = self.work_dirpath / "output" / dirname / str(instance_index)
        self.make_dir(res)

        return res

    def make_dir(self, dirname, clobber=None):
        if clobber is None:
            clobber = self.clobber

        dirpath = path(dirname)

        if clobber:
            # just always try to delete it
            try:
                dirpath.rmtree()
            except OSError, err:
                if err.errno != ENOENT:
                    raise
        try:
            dirpath.makedirs()
        except OSError, err:
            # if the error is because directory exists, but it's
            # empty, then do nothing
            if (err.errno != EEXIST or not dirpath.isdir() or
                dirpath.listdir()):
                raise

    def make_subdir(self, subdirname):
        self.make_dir(self.work_dirpath / subdirname)

    def make_subdirs(self, subdirnames):
        for subdirname in subdirnames:
            self.make_subdir(subdirname)

    def make_obs_filelistpath(self, ext):
        return make_filelistpath(self.obs_dirpath, ext)

    def save_resource(self, resname, subdirname=""):
        orig_filename = data_filename(resname)

        orig_filepath = path(orig_filename)
        dirpath = self.work_dirpath / subdirname

        orig_filepath.copy(dirpath)
        return dirpath / orig_filepath.name

    def save_include(self):
        # XXX: can this become a memoized property?
        # We may need to write the file even if it is specified
        aux_dirpath = self.work_dirpath / SUBDIRNAME_AUX

        self.gmtk_include_filename, self.gmtk_include_filename_is_new = \
            IncludeSaver(self)(self.gmtk_include_filename, aux_dirpath,
                               self.clobber)

    def subset_metadata_attr(self, genome, name, reducer=sum):
        attr = getattr(genome, name)

        tied_track_indexes_list = self.tied_track_indexes_list
        shape = len(tied_track_indexes_list)
        subset_array = empty(shape, attr.dtype)
        for index, tied_track_indexes in enumerate(tied_track_indexes_list):
            subset_array[index] = reducer(attr[tied_track_indexes])

        if __debug__:
            track_indexes = self.track_indexes
            if len(self.head_trackname_list) == len(track_indexes):
                # ensure that the results are the same as the old method
                assert len(subset_array) == len(track_indexes)
                assert (subset_array == attr[track_indexes]).all()

        setattr(self, name, subset_array)

    def subset_metadata(self, genome):
        """
        limits all the metadata attributes to only tracks that are used
        """
        subset_metadata_attr = self.subset_metadata_attr
        subset_metadata_attr(genome, "mins", min)
        subset_metadata_attr(genome, "maxs", max)
        subset_metadata_attr(genome, "sums")
        subset_metadata_attr(genome, "sums_squares")
        subset_metadata_attr(genome, "num_datapoints")

    def save_input_master(self, instance_index=None, new=False):
        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        self.input_master_filename, input_master_filename_is_new = \
            InputMasterSaver(self)(input_master_filename, self.params_dirpath,
                                   self.clobber, instance_index)

    def load_supervision(self):
        supervision_type = self.supervision_type
        if supervision_type == SUPERVISION_UNSUPERVISED:
            return

        assert supervision_type == SUPERVISION_SEMISUPERVISED

        supervision_labels = defaultdict(list)
        supervision_coords = defaultdict(list)

        with open(self.supervision_filename) as supervision_file:
            for datum in read_native(supervision_file):
                chrom = datum.chrom
                start = datum.chromStart
                end = datum.chromEnd

                # XXX change check_overlapping_supervision labels to do something
                # that's not supervision-specific
                check_overlapping_supervision_labels(start, end, chrom,
                                                     supervision_coords)

                supervision_coords[chrom].append((start, end))
                supervision_labels[chrom].append(int(datum.name))

        max_supervision_label = max(max(labels)
                                    for labels
                                    in supervision_labels.itervalues())

        self.supervision_coords = supervision_coords
        self.supervision_labels = supervision_labels

        self.include_tracknames.append("supervisionLabel")
        self.card_supervision_label = (max_supervision_label + 1 +
                                       SUPERVISION_LABEL_OFFSET)

    # XXXmax
    def load_virtual_evidence(self):
        if not self.virtual_evidence:
            return

        virtual_evidence_coords = defaultdict(list)
        virtual_evidence_evidence = defaultdict(list)

        with open(self.virtual_evidence_filename, "r") as virtual_evidence_file:
            for datum in read_bed3(virtual_evidence_file):
                chrom = datum.chrom
                start = int(datum.chromStart)
                end = int(datum.chromEnd)
                try:
                    evidence = map(permissive_log, map(float, datum._words[3:]))
                except ValueError:
                    print >>sys.stderr, """
                    Error reading virtual evidence file: %s
                    Virtual evidence values must be floats
                    """ % self.virtual_evidence_filename
                    raise

                if self.num_worlds != 1:
                    chrom_split = chrom.split(".", 1)
                    if len(chrom_split) != 2:
                        raise ValueError("""
                                         Error reading virtual evidence file: %s
                                         When running with concatenated tracks, the chrom field
                                         must have the format <world>.<chrom>
                                         """ % (self.virtual_evidence_filename))
                    world = int(chrom_split[0])
                    chrom = chrom_split[1]
                else:
                    world = 0

                check_overlapping_supervision_labels(start, end, (world, chrom),
                                                     virtual_evidence_coords)

                virtual_evidence_coords[(world, chrom)].append((start, end))


                assert(len(evidence) == self.num_segs)

                virtual_evidence_evidence[(world, chrom)].append(evidence)

        def make_obs_iter():
            for window_index, (world, chrom, start, end) in enumerate(self.windows):
                resolution = self.resolution

                def make_window_iter():
                    window_overlaps = find_overlaps_include(start, end,
                                                            virtual_evidence_coords[(world, chrom)],
                                                            virtual_evidence_evidence[(world, chrom)])
                    window_overlaps = sorted(window_overlaps, key=lambda overlap: overlap[0])

                    def round_up_to_resolution(pos, start, resolution):
                        return ceildiv(pos-start,resolution)*resolution + start
                    def round_down_to_resolution(pos, start, resolution):
                        return int((pos-start)/resolution)*resolution + start

                    cur = start
                    overlap_index = 0
                    evidence_fmt = "%sf" % self.num_segs
                    uniform_evidence = [log(float(1)/self.num_segs) for i in range(self.num_segs)]
                    while True:
                        # after the last overlap, just set the rest of the window to
                        # uniform and then break
                        if overlap_index >= len(window_overlaps):
                            overlap_start = end
                            # for the last frame, round up to the resolution
                            overlap_start = round_up_to_resolution(overlap_start, start, resolution)
                            #overlap_start += resolution - ((overlap_start - start) % resolution)
                            overlap_end = overlap_start
                        else:
                            overlap_start, overlap_end, evidence = window_overlaps[overlap_index]
                            # lock overlap to resolution by extending the overlap if need be
                            overlap_start = round_down_to_resolution(overlap_start, start, resolution)
                            #overlap_start -= (overlap_start - start) % resolution
                            overlap_end = round_up_to_resolution(overlap_end, start, resolution)
                            #overlap_end += resolution - ((overlap_end - start) % resolution)

                        try:
                            #assert ((end - start) % resolution == 0)
                            assert ((overlap_start - start) % resolution == 0)
                            assert ((overlap_end - start) % resolution == 0)
                            assert ((overlap_end - overlap_start) % resolution == 0)
                            assert ((overlap_start - cur) % resolution == 0)
                        except:
                            raise

                        # set the region from cur to the next window as uniform
                        for i in range(int((overlap_start - cur) / resolution)):
                            yield uniform_evidence
                        cur = overlap_start

                        if overlap_index >= len(window_overlaps):
                            break

                        # set the overlapping region according to the virtual evidence file
                        for i in range(int((overlap_end - cur) / resolution)):
                            yield evidence

                        cur = overlap_end
                        overlap_index += 1
                yield make_window_iter()


        write_virtual_evidence(make_obs_iter(), self.virtual_evidence_dirpath,
                               self.windows, self.num_segs)

    def load_measure_prop(self):
        def make_obs_iter():
            ve_line = [log(float(1)/(self.num_segs)) for i in range(self.num_segs)]
            for window_index, (world, chrom, start, end) in enumerate(self.windows):
                num_frames = ceildiv(end-start, self.resolution)
                yield (ve_line for frame_index in range(num_frames))

        write_virtual_evidence(make_obs_iter(),
                               self.uniform_ve_dirname,
                               self.windows,
                               self.num_segs)


    def save_structure(self):
        self.structure_filename, _ = \
            StructureSaver(self)(self.structure_filename, self.work_dirname,
                                 self.clobber)

    def save_observations_params(self):
        # XXX: these expect different filepaths
        assert not ((self.identify or self.posterior) and self.train)

        self.load_supervision()

        # need to open Genomedata archive first in order to determine
        # self.tracknames and self.num_tracks

        GenomeClass = (FilesGenome if self.file_tracks else Genome)
        genomedataarg = (self.track_specs if self.file_tracks else self.genomedataname)
        if self.file_tracks and self.num_worlds > 1:
            raise NotImplementedError # TODO implement concat handling within FilesGenome
        with GenomeClass(genomedataarg) as genome:
            self.set_tracknames(genome)

            observations = Observations(self)
            observations.locate_windows(genome)

            self.windows = observations.windows
            self.subset_metadata(genome) # XXX: does this need to be done before save()?

            observations.save(genome)

        self.float_filepaths = observations.float_filepaths
        self.int_filepaths = observations.int_filepaths

        self.load_virtual_evidence()

        if self.train:
            self.set_log_likelihood_filenames()

        self.save_include()
        self.set_params_filename()
        self.save_structure()

        self.load_measure_prop()

    def copy_results(self, name, src_filename, dst_filename):
        if dst_filename:
            copy2(src_filename, dst_filename)
        else:
            dst_filename = src_filename

        setattr(self, name, dst_filename)

    def prog_factory(self, prog):
        """
        allows dry_run
        """
        # XXX: this poisons a global variable
        prog.dry_run = self.dry_run

        return prog

    def make_acc_filename(self, instance_index, window_index):
        return self.make_filename(PREFIX_ACC, instance_index, window_index,
                                  EXT_BIN, subdirname=SUBDIRNAME_ACC)

    def make_posterior_filename(self, window_index):
        return self.make_filename(PREFIX_POSTERIOR, window_index, EXT_BED, EXT_GZ,
                                  subdirname=SUBDIRNAME_POSTERIOR)

    def make_job_name_train(self, instance_index, round_index, window_index):
        return "%s%d.%d.%s.%s.%s" % (PREFIX_JOB_NAME_TRAIN, instance_index,
                                     round_index, window_index,
                                     self.work_dirpath.name, self.uuid)

    def make_job_name_identify(self, prefix, window_index):
        return "%s%d.%s.%s" % (prefix, window_index, self.work_dirpath.name,
                               self.uuid)

    def make_gmtk_kwargs(self):
        """
        shared args to gmtkEMtrain, gmtkViterbi, gmtkJT
        """
        res = dict(strFile=self.structure_filename,
                   verbosity=self.verbosity,
                   island=ISLAND,
                   componentCache=COMPONENT_CACHE,
                   deterministicChildrenStore=DETERMINISTIC_CHILDREN_STORE,
                   jtFile=self.jt_info_filename,
                   seed="T", # XXX
                   obsNAN=True)

        if ISLAND:
            res["base"] = ISLAND_BASE
            res["lst"] = ISLAND_LST

        if HASH_LOAD_FACTOR is not None:
            res["hashLoadFactor"] = HASH_LOAD_FACTOR

        # XXX: dinucleotide-only won't work, because it has no float data
        assert self.float_filelistpath and self.num_tracks
        if self.float_filelistpath:
            res.update(of1=self.float_filelistpath,
                       fmt1="binary",
                       nf1=self.num_tracks,
                       ni1=0,
                       iswp1=SWAP_ENDIAN)

        if self.int_filelistpath and self.num_int_cols:
            res.update(of2=self.int_filelistpath,
                       fmt2="binary",
                       nf2=0,
                       ni2=self.num_int_cols,
                       iswp2=SWAP_ENDIAN)

        return res

    def window_lens_sorted(self, reverse=True):
        """
        yields (window_index, window_mem_usage)

        if reverse: sort windows by decreasing size, so the most
        difficult windows are dropped in the queue first
        """
        window_lens = self.window_lens

        # XXX: use key=itemgetter(2) and enumerate instead of this silliness
        zipper = sorted(izip(window_lens, count()), reverse=reverse)

        # XXX: use itertools instead of a generator
        for window_len, window_index in zipper:
            yield window_index, window_len

    def log_cmdline(self, cmdline, args=None):
        if args is None:
            args = cmdline

        _log_cmdline(self.cmdline_short_file, cmdline)
        _log_cmdline(self.cmdline_long_file, args)

    def calc_tmp_usage(self, num_frames, prog):
        if prog in TMP_OBS_PROGS:
            tmp_usage_obs = num_frames * self.num_tracks * SIZEOF_FRAME_TMP
        else:
            tmp_usage_obs = 0

        return tmp_usage_obs + TMP_USAGE_BASE

    def queue_gmtk(self, prog, kwargs, job_name, num_frames,
                   output_filename=None, prefix_args=[]):
        gmtk_cmdline = prog.build_cmdline(options=kwargs)

        if prefix_args:
            # remove the command name itself from the passed arguments
            # XXX: this is ugly
            args = prefix_args + gmtk_cmdline[1:]
        else:
            args = gmtk_cmdline


        # this doesn't include use of segway-wrapper, which takes the
        # memory usage as an argument, and may be run multiple times
        self.log_cmdline(gmtk_cmdline, args)

        # XXX This code fixes the really strange nondeterministic
        # segfault bug.  I have no idea why it's necessary
        args = map(str, args)
        args = map(lambda s: s.replace("%s", "SENTINEL_PERCENT_SIGN"), args)

        if self.dry_run:
            return None

        session = self.session
        job_tmpl = session.createJobTemplate()

        job_tmpl.jobName = job_name
        job_tmpl.remoteCommand = ENV_CMD
        job_tmpl.args = map(str, args)


        # this is going to cause problems on heterogeneous systems
        environment = environ.copy()
        try:
            # this causes errors
            del environment["PYTHONINSPECT"]
        except KeyError:
            pass
        job_tmpl.jobEnvironment = environment

        if output_filename is None:
            output_filename = self.output_dirpath / job_name
        error_filename = self.error_dirpath / job_name

        job_tmpl.blockEmail = True

        job_tmpl.nativeSpecification = make_native_spec(*self.user_native_spec)

        set_cwd_job_tmpl(job_tmpl)

        tmp_usage = self.calc_tmp_usage(num_frames, prog)

        job_tmpl_factory = JobTemplateFactory(job_tmpl,
                                              tmp_usage,
                                              self.mem_usage_progression,
                                              output_filename,
                                              error_filename)

        mem_usage_key = (prog.prog, self.num_segs, num_frames)

        # XXXopt: should be able to calculate exactly the first
        # trial_index to start with, need to at least be able to load
        # data into RAM

        # XXX: should not have MemoryErrors

        return RestartableJob(session, job_tmpl_factory, self.global_mem_usage,
                              mem_usage_key)

    def queue_train(self, instance_index, round_index, window_index, num_frames=0,
                    **kwargs):
        """
        this calls Runner.queue_gmtk()

        if num_frames is not specified, then it is set to 0, where
        everyone will share their min/max memory usage. Used for calls from queue_train_bundle()
        """
        kwargs["inputMasterFile"] = self.input_master_filename

        name = self.make_job_name_train(instance_index, round_index, window_index)

        return self.queue_gmtk(self.train_prog, kwargs, name, num_frames)

    def queue_train_parallel(self, input_params_filename, instance_index,
                             round_index, **kwargs):
        kwargs["cppCommandOptions"] = self.make_cpp_options("train", instance_index, round_index,
                                                            input_params_filename=input_params_filename)

        res = RestartableJobDict(self.session, self.job_log_file)

        make_acc_filename_custom = partial(self.make_acc_filename, instance_index)

        for window_index, window_len in self.window_lens_sorted():
            acc_filename = make_acc_filename_custom(window_index)
            kwargs_window = dict(trrng=window_index, storeAccFile=acc_filename,
                                 **kwargs)

            # -dirichletPriors T only on the first window
            kwargs_window["dirichletPriors"] = (window_index == 0)

            if self.is_in_reversed_world(window_index):
                kwargs_window["gpr"] = REVERSE_GPR

            num_frames = self.window_lens[window_index]

            restartable_job = self.queue_train(instance_index, round_index,
                                               window_index, num_frames,
                                               **kwargs_window)
            res.queue(restartable_job)

        return res

    def queue_train_bundle(self, input_params_filename, output_params_filename,
                           instance_index, round_index, **kwargs):
        """bundle step: take parallel accumulators and combine them
        """
        acc_filename = self.make_acc_filename(instance_index,
                                              GMTK_INDEX_PLACEHOLDER)

        cpp_options = self.make_cpp_options("train", instance_index, round_index,
                                            input_params_filename=input_params_filename,
                                            output_params_filename=output_params_filename)

        last_window = self.num_windows - 1

        kwargs = dict(outputMasterFile=self.output_master_filename,
                      cppCommandOptions=cpp_options,
                      trrng="nil",
                      loadAccRange="0:%s" % last_window,
                      loadAccFile=acc_filename,
                      **kwargs)

        restartable_job = self.queue_train(instance_index, round_index,
                                           NAME_BUNDLE_PLACEHOLDER, **kwargs)

        res = RestartableJobDict(self.session, self.job_log_file)
        res.queue(restartable_job)

        return res

    def get_posterior_clique_print_ranges(self):
        res = {}

        for clique, clique_index in self.posterior_clique_indices.iteritems():
            range_str = "%d:%d" % (clique_index, clique_index)
            res[clique + "CliquePrintRange"] = range_str

        return res

    def set_triangulation_filename(self, num_segs=None, num_subsegs=None):
        if num_segs is None:
            num_segs = self.num_segs

        if num_subsegs is None:
            num_subsegs = self.num_subsegs

        if (self.triangulation_filename_is_new
            or not self.triangulation_filename):
            self.triangulation_filename_is_new = True

            structure_filebasename = path(self.structure_filename).name
            triangulation_filebasename = \
                extjoin(structure_filebasename, str(num_segs),
                        str(num_subsegs), EXT_TRIFILE)

            self.triangulation_filename = (self.triangulation_dirpath
                                           / triangulation_filebasename)


    def run_triangulate_single(self, num_segs, num_subsegs=None):
        # print >>sys.stderr, "running triangulation"
        prog = self.prog_factory(TRIANGULATE_PROG)

        self.set_triangulation_filename(num_segs, num_subsegs)

        cpp_options = self.make_cpp_options("triangulate")
        kwargs = dict(strFile=self.structure_filename,
                      cppCommandOptions=cpp_options,
                      outputTriangulatedFile=self.triangulation_filename,
                      verbosity=self.verbosity)

        # XXX: need exist/clobber logic here
        # XXX: repetitive with queue_gmtk
        self.log_cmdline(prog.build_cmdline(options=kwargs))

        prog(**kwargs)

    def run_triangulate(self):
        for num_segs in self.num_segs_range:
            self.run_triangulate_single(num_segs)

    def run_train_round(self, instance_index, round_index, **kwargs):
        """
        returns None: normal
        returns not None: abort
        """
        last_params_filename = self.last_params_filename
        curr_params_filename = extjoin(self.params_filename, str(round_index))

        if self.measure_prop_graph_filepath and (round_index > 0):
            for i in range(self.measure_prop_num_iters):
                mp_round_index = "%s_%s" % (round_index, i)
                self.mp_runner.update(instance_index, mp_round_index, last_params_filename)

        restartable_jobs = \
            self.queue_train_parallel(last_params_filename, instance_index,
                                      round_index, **kwargs)
        restartable_jobs.wait()

        restartable_jobs = \
            self.queue_train_bundle(last_params_filename, curr_params_filename,
                                    instance_index, round_index,
                                    llStoreFile=self.log_likelihood_filename,
                                    **kwargs)
        restartable_jobs.wait()

        self.last_params_filename = curr_params_filename

    def run_train_instance(self):
        self.set_triangulation_filename()

        # make new files if there is more than one instance
        new = self.instance_make_new_params

        instance_index = self.instance_index
        self.set_log_likelihood_filenames(instance_index, new)
        self.set_params_filename(instance_index, new)

        # get previous (or initial) values
        last_log_likelihood, log_likelihood, round_index = \
            self.recover_train_instance()

        if round_index == 0:
            # if round > 0, this is set by self.recover_train_instance()
            self.save_input_master(instance_index, new)

        if self.measure_prop_graph_filepath:
            self.mp_runner = MeasurePropRunner(self)
            self.mp_runner.load(instance_index)
            last_mp_terms = [float("nan"), float("nan"), float("nan")] # XXX
            mp_terms = [float("nan"), float("nan"), float("nan")] # XXX
        else:
            last_mp_terms = None
            mp_terms = None

        kwargs = dict(objsNotToTrain=self.dont_train_filename,
                      maxEmIters=1,
                      lldp=LOG_LIKELIHOOD_DIFF_FRAC*100.0,
                      triFile=self.triangulation_filename,
                      **self.make_gmtk_kwargs())

        if self.dry_run:
            self.run_train_round(self.instance_index, round_index, **kwargs)
            return Results(None, None, None, None, None, None, None)

        return self.progress_train_instance(last_log_likelihood,
                                            log_likelihood,
                                            last_mp_terms, mp_terms,
                                            round_index, kwargs)

    def progress_train_instance(self, last_log_likelihood, log_likelihood,
                                last_mp_terms, mp_terms,
                                round_index, kwargs):
        last_objective = objective_value(last_log_likelihood, last_mp_terms)
        objective = objective_value(log_likelihood, mp_terms)
        while (round_index < self.max_em_iters and
               is_training_progressing(last_objective, objective)):
            self.run_train_round(self.instance_index, round_index, **kwargs)

            last_log_likelihood = log_likelihood
            last_mp_terms = mp_terms
            last_objective = objective

            log_likelihood = self.load_log_likelihood()
            if (self.measure_prop_graph_filepath):
                if (round_index > 0):
                    last_mp_round_index = "%s_%s" % (round_index, self.measure_prop_num_iters-1)
                    mp_terms = self.load_measure_prop_objective(last_mp_round_index)
                else:
                    mp_terms = [float("nan"), float("nan"), float("nan")]
            else:
                mp_terms = None
            objective = objective_value(log_likelihood, mp_terms)
            round_index += 1

        # log_likelihood, num_segs and a list of src_filenames to save
        return Results(self.instance_index, log_likelihood, mp_terms, self.num_segs, self.input_master_filename,
                       self.last_params_filename, self.log_likelihood_filename)

    def save_train_options(self):
        filename = self.make_filename(TRAIN_FILEBASENAME)

        with open(filename, "w") as tabfile:
            writer = ListWriter(tabfile)
            writer.writerow(TRAIN_FIELDNAMES)

            for name, typ in sorted(TRAIN_OPTION_TYPES.iteritems()):
                value = getattr(self, name)
                if isinstance(typ, list):
                    for item in value:
                        writer.writerow([name, item])
                else:
                    writer.writerow([name, value])

    def load_train_options(self, traindirname):
        """
        load options from training and convert to appropriate type
        """
        filename = path(traindirname) / TRAIN_FILEBASENAME

        with open(filename) as tabfile:
            reader = DictReader(tabfile)

            for row in reader:
                name = row["name"]
                value = row["value"]

                typ = TRAIN_OPTION_TYPES[name]
                if isinstance(typ, list):
                    assert len(typ) == 1
                    item_typ = typ[0]
                    getattr(self, name).append(item_typ(value))
                else:
                    setattr(self, name, typ(value))

        if self.params_filename is not None:
            self.params_filenames = [self.params_filename]

    def setup_train(self):
        """
        return value: dst_filenames
        """
        assert self.num_instances >= 1

        # save the destination file for input_master as we will be
        # generating new input masters for each start

        # must be before file creation. Otherwise
        # input_master_filename_is_new will be wrong
        input_master_filename, input_master_filename_is_new = \
            InputMasterSaver(self)(self.input_master_filename,
                                   self.params_dirpath, self.clobber)

        self.input_master_filename = input_master_filename

        # should I make new parameters in each instance?
        if not self.instance_make_new_params:
            self.save_input_master()

        ## save file locations to tab-delimited file
        self.save_train_options()

        if not input_master_filename_is_new:
            # do not overwrite existing file
            input_master_filename = None

        return [input_master_filename, self.params_filename,
                self.log_likelihood_filename]

    def get_thread_run_func(self):
        if len(self.num_segs_range) > 1 or self.num_instances > 1:
            return self.run_train_multithread
        else:
            return self.run_train_singlethread

    def finish_train(self, instance_params, dst_filenames):
        if self.instance_make_new_params:
            self.proc_train_results(instance_params, dst_filenames)
        elif not self.dry_run:
            # only one instance
            assert len(instance_params) == 1
            last_params_filename = instance_params[0].params_filename
            copy2(last_params_filename, self.params_filename)

            # always overwrite params.params
            copy2(last_params_filename, self.make_params_filename())

    def run_train(self):
        dst_filenames = self.setup_train()

        run_train_func = self.get_thread_run_func()

        ## this is where the actual training takes place
        instance_params = run_train_func(self.num_segs_range)

        self.finish_train(instance_params, dst_filenames)

    def run_train_singlethread(self, num_segs_range):
        # having a single-threaded version makes debugging much easier
        with Session() as session:
            self.session = session
            self.instance_index = 0
            res = [self.run_train_instance()]

        self.session = None

        return res

    def run_train_multithread(self, num_segs_range):
        seg_instance_indexes = xrange(self.num_instances)
        enumerator = enumerate(product(num_segs_range, seg_instance_indexes))

        # ensure memoization before threading
        self.triangulation_dirpath
        self.jt_info_filename
        self.include_coords
        self.exclude_coords
        self.card_seg_countdown
        self.obs_dirpath
        self.float_filelistpath
        self.int_filelistpath
        self.float_tabfilepath
        self.gmtk_include_filename_relative
        self.means
        self.vars
        self.dont_train_filename
        self.output_master_filename
        self.params_dirpath
        self.window_lens
        self.use_dinucleotide
        self.num_int_cols
        self.train_prog
        self.posterior_dirname

        threads = []
        with Session() as session:
            try:
                for instance_index, (num_seg, seg_instance_index) in enumerator:
                    # print >>sys.stderr, (
                    #    "instance_index %s, num_seg %s, seg_instance_index %s"
                    #    % (instance_index, num_seg, seg_instance_index))
                    thread = TrainThread(self, session, instance_index,
                                         num_seg)
                    thread.start()
                    threads.append(thread)

                    # let all of one thread's jobs drop in the queue
                    # before you do the next one
                    # XXX: using some sort of semaphore would be better
                    # XXX: using a priority option to the system would be best
                    sleep(THREAD_START_SLEEP_TIME)

                # list of tuples(log_likelihood, input_master_filename,
                #                params_filename)
                instance_params = []
                for thread in threads:
                    while thread.isAlive():
                        # XXX: KeyboardInterrupts only occur if there is a
                        # timeout specified here. Is this a Python bug?
                        thread.join(JOIN_TIMEOUT)

                    # this will get AttributeError if the thread failed and
                    # therefore did not set thread.result
                    try:
                        thread_result = thread.result
                    except AttributeError:
                        raise AttributeError("""\
Training instance %s failed. See previously printed error for reason.
Final params file will not be written. Rerun the instance or use segway-winner
to find the winning instance anyway.""" % thread.instance_index)
                    else:
                        instance_params.append(thread_result)
            except KeyboardInterrupt:
                self.interrupt_event.set()
                for thread in threads:
                    thread.join()

                raise

        return instance_params

    def proc_train_results(self, instance_params, dst_filenames):
        if self.dry_run:
            return

        # finds the min by info_criterion (maximize log_likelihood)
        max_params = sorted(instance_params, cmp=results_objective_cmp, reverse=True)[0]

        # write winning instance index
        with open(self.winner_filename, "w") as winner_f:
            print >>winner_f, max_params.instance_index

        self.num_segs = max_params.num_segs
        self.set_triangulation_filename()

        src_filenames = max_params[OFFSET_FILENAMES:]

        if None in src_filenames:
            raise ValueError("all training instances failed")

        assert LEN_TRAIN_ATTRNAMES == len(src_filenames) == len(dst_filenames)

        zipper = zip(TRAIN_ATTRNAMES, src_filenames, dst_filenames)
        for name, src_filename, dst_filename in zipper:
            self.copy_results(name, src_filename, dst_filename)

    def recover_filename(self, resource):
        instance_index = self.instance_index

        # only want "input.master" not "input.0.master" if there is
        # only one instance
        if (not self.instance_make_new_params
            and resource == InputMasterSaver.resource_name):
            instance_index = None

        old_filename = make_default_filename(resource,
                                             self.recover_params_dirpath,
                                             instance_index)

        new_filename = make_default_filename(resource, self.params_dirpath,
                                             instance_index)

        path(old_filename).copy2(new_filename)
        return new_filename

    def recover_train_instance(self):
        """
        returns last_log_likelihood, log_likelihood, round_index
        -inf, -inf, 0 if there is no recovery--this is also used to set initial
        values
        """
        last_log_likelihood = -inf
        log_likelihood = -inf
        final_round_index = 0

        if self.recover_dirpath:
            instance_index = self.instance_index
            recover_dirname = self.recover_dirname

            self.input_master_filename = \
                self.recover_filename(InputMasterSaver.resource_name)

            recover_log_likelihood_tab_filename = \
                self.make_log_likelihood_tab_filename(instance_index,
                                                      recover_dirname)

            with open(recover_log_likelihood_tab_filename) as log_likelihood_tab_file:
                log_likelihoods = [float(line.rstrip())
                                   for line in log_likelihood_tab_file.readlines()]

            final_round_index = len(log_likelihoods)
            if final_round_index > 0:
                log_likelihood = log_likelihoods[-1]
            if final_round_index > 1:
                last_log_likelihood = log_likelihoods[-2]

            path(recover_log_likelihood_tab_filename).copy2(self.log_likelihood_tab_filename)

            old_params_filename = self.make_params_filename(instance_index,
                                                            recover_dirname)
            new_params_filename = self.params_filename
            for round_index in xrange(final_round_index):
                old_curr_params_filename = extjoin(old_params_filename,
                                                   str(round_index))
                new_curr_params_filename = extjoin(new_params_filename,
                                                   str(round_index))

                path(old_curr_params_filename).copy2(new_curr_params_filename)

            self.last_params_filename = new_curr_params_filename

        return last_log_likelihood, log_likelihood, final_round_index

    def recover_viterbi_window(self, window_index):
        """
        returns False if no recovery
                True if recovery
        """
        recover_filenames = self.recover_viterbi_filenames
        if not recover_filenames:
            return False

        recover_filename = recover_filenames[window_index]
        try:
            with open(recover_filename) as oldfile:
                lines = oldfile.readlines()
        except IOError, err:
            if err.errno == ENOENT:
                return False
            else:
                raise

        window = self.windows[window_index]
        window_chrom = window.chrom

        # XXX: duplicative
        row, line_coords = parse_bed4(lines[0])
        (line_chrom, line_start, line_end, seg) = line_coords
        if line_chrom != window_chrom or int(line_start) != window.start:
            return False

        row, line_coords = parse_bed4(lines[-1])
        (line_chrom, line_start, line_end, seg) = line_coords
        if line_chrom != window_chrom or int(line_end) != window.end:
            return False

        # copy the old filename to where the job's output would have
        # landed
        path(recover_filename).copy2(self.viterbi_filenames[window_index])

        print >>sys.stderr, "window %d already complete" % window_index

        return True

    def queue_identify(self, restartable_jobs, window_index, params_filename,
                       prefix_job_name, prog, kwargs, output_filenames):
        prog = self.prog_factory(prog)
        job_name = self.make_job_name_identify(prefix_job_name, window_index)
        output_filename = output_filenames[window_index]

        kwargs = self.get_identify_kwargs(window_index, kwargs, params_filename)

        if prog == VITERBI_PROG:
            kind = "viterbi"
        else:
            kind = "posterior"

        # "0" or "1"
        is_reverse = str(int(self.is_in_reversed_world(window_index)))

        window = self.windows[window_index]
        float_filepath = self.float_filepaths[window_index]
        int_filepath = self.int_filepaths[window_index]

        track_indexes = self.world_track_indexes[window.world]
        track_indexes_text = ",".join(map(str, track_indexes))

        genomedataarg = (FILE_TRACKS_SENTINEL if self.file_tracks else self.genomedataname)
        track_indexes_text_arg = (",".join(self.track_specs) if self.file_tracks else track_indexes_text)

        prefix_args = [find_executable("segway-task"), "run", kind,
                       output_filename, window.chrom,
                       window.start, window.end, self.resolution, is_reverse,
                       self.num_segs, genomedataarg, float_filepath,
                       int_filepath, self.distribution,
                       track_indexes_text_arg]
        output_filename = None

        num_frames = self.window_lens[window_index]

        restartable_job = self.queue_gmtk(prog, kwargs, job_name,
                                          num_frames,
                                          output_filename=output_filename,
                                          prefix_args=prefix_args)

        restartable_jobs.queue(restartable_job)

    def get_identify_kwargs(self, window_index, extra_kwargs, params_filename):
        cpp_command_options = self.make_cpp_options("identify", window_index=window_index,
                                                    input_params_filename=params_filename)

        res = dict(inputMasterFile=self.input_master_filename,
                   cppCommandOptions=cpp_command_options,
                   cliqueTableNormalize="0.0",
                   **self.make_gmtk_kwargs())

        if self.is_in_reversed_world(window_index):
            res["gpr"] = REVERSE_GPR

        res.update(extra_kwargs)

        return res

    def is_in_reversed_world(self, window_index):
        return self.windows[window_index].world in self.reverse_worlds

    # Used by run_identify_posterior and MeasurePropRunner.update
    def run_identify_posterior_jobs(self, identify, posterior,
                                    viterbi_filenames,
                                    posterior_filenames,
                                    params_filename=None,
                                    instance_index="identify",
                                    round_index="identify"):

        if params_filename is None:
            params_filename = self.params_filename

        # -: standard output, processed by segway-task
        viterbi_kwargs = dict(triFile=self.triangulation_filename,
                              pVitRegexFilter="^seg$",
                              pVitValsFile="-")

        posterior_kwargs = dict(triFile=self.posterior_triangulation_filename,
                                jtFile=self.posterior_jt_info_filename,
                                doDistributeEvidence=True,
                                **self.get_posterior_clique_print_ranges())

        session = self.session
        restartable_jobs = RestartableJobDict(session, self.job_log_file)
        if self.measure_prop_graph_filepath:
            measure_prop_ve_list_filenames = self.make_measure_prop_ve_window_list_filenames(instance_index, round_index)

        for window_index, window_len in self.window_lens_sorted():

            if self.measure_prop_graph_filepath:
                measure_prop_ve_list_filename = measure_prop_ve_list_filenames[window_index]
            virtual_evidence_ve_list_filename = self.virtual_evidence_ve_window_list_filenames[window_index]

            queue_identify_custom = partial(self.queue_identify,
                                            restartable_jobs, window_index,
                                            params_filename)

            if (identify
                and not self.recover_viterbi_window(window_index)):
                queue_identify_custom(PREFIX_JOB_NAME_VITERBI,
                                      VITERBI_PROG, viterbi_kwargs,
                                      viterbi_filenames)

            if posterior:
                queue_identify_custom(PREFIX_JOB_NAME_POSTERIOR,
                                      POSTERIOR_PROG, posterior_kwargs,
                                      posterior_filenames)

        # XXX: ask on DRMAA mailing list--how to allow
        # KeyboardInterrupt here?

        if self.dry_run:
            return

        restartable_jobs.wait()


    def run_identify_posterior(self):
        self.instance_index = "identify"

        ## setup files
        if not self.input_master_filename or not path(self.input_master_filename).isfile():
            warn("Input master not specified. Generating.")
            self.make_subdir(SUBDIRNAME_PARAMS)
            self.save_input_master()

        viterbi_filenames = self.viterbi_filenames
        posterior_filenames = self.posterior_filenames

        if self.measure_prop_graph_filepath:
            self.mp_runner = MeasurePropRunner(self)
            self.mp_runner.load("identify")

        # XXX: kill submitted jobs on exception
        with Session() as session:
            self.session = session

            if self.measure_prop_graph_filepath:
                for i in range(self.measure_prop_num_iters):
                    instance_index = "identify"
                    round_index = "identify_%s" % i
                    self.mp_runner.update(instance_index, round_index, self.params_filename)
            else:
                instance_index = "identify"
                round_index = "identify"

            self.run_identify_posterior_jobs(self.identify, self.posterior,
                                             viterbi_filenames, posterior_filenames,
                                             instance_index=instance_index,
                                             round_index=round_index)

        for world in xrange(self.num_worlds):
            if self.identify:
                IdentifySaver(self)(world)

            if self.posterior:
                PosteriorSaver(self)(world)

    def make_script_filename(self, prefix):
        return self.make_filename(prefix, EXT_SH, subdirname=SUBDIRNAME_LOG)

    def make_run_msg(self):
        now = datetime.now()
        pkg_desc = working_set.find(Requirement.parse(__package__))
        run_msg = "## %s run %s at %s" % (pkg_desc, self.uuid, now)

        cmdline_top_filename = self.make_script_filename(PREFIX_CMDLINE_TOP)

        with open(cmdline_top_filename, "w") as cmdline_top_file:
            print >>cmdline_top_file, run_msg
            print >>cmdline_top_file
            print >>cmdline_top_file, "cd %s" % maybe_quote_arg(path.getcwd())
            print >>cmdline_top_file, cmdline2text()

        return run_msg

    def run(self):
        """
        main run, after dirname is specified

        this is exposed so that it can be overriden in a subclass

        opens log files, saves parameters, and calls main function
        run_train() or run_identify_posterior()
        """
        # XXXopt: use binary I/O to gmtk rather than ascii for parameters

        self.interrupt_event = Event()

        ## start log files
        self.make_subdir(SUBDIRNAME_LOG)
        run_msg = self.make_run_msg()

        cmdline_short_filename = self.make_script_filename(PREFIX_CMDLINE_SHORT)
        cmdline_long_filename = self.make_script_filename(PREFIX_CMDLINE_LONG)
        job_log_filename = self.make_filename(PREFIX_JOB_LOG, EXT_TAB,
                                              subdirname=SUBDIRNAME_LOG)

        self.make_subdirs(SUBDIRNAMES_EITHER)

        if self.train:
            self.make_subdirs(SUBDIRNAMES_TRAIN)

        self.save_observations_params()

        with open(cmdline_short_filename, "w") as self.cmdline_short_file:
            with open(cmdline_long_filename, "w") as self.cmdline_long_file:
                print >>self.cmdline_short_file, run_msg
                print >>self.cmdline_long_file, run_msg

                self.run_triangulate()

                with open(job_log_filename, "w") as self.job_log_file:
                    print >>self.job_log_file, "\t".join(JOB_LOG_FIELDNAMES)

                    if self.train:
                        self.run_train()

                    if self.identify or self.posterior:
                        if self.supervision_filename:
                            raise NotImplementedError # XXX

                        if not self.dry_run:
                            # resave now that num_segs is determined,
                            # in case you tested multiple num_segs
                            self.save_include()

                        if (self.posterior and (self.recover_dirname
                                                or self.num_worlds != 1)):
                            print >>sys.stderr, "Running posterior even though it's listed as not implemented!"
                            #raise NotImplementedError # XXX

                        self.run_identify_posterior()

    def __call__(self, *args, **kwargs):
        # XXX: register atexit for cleanup_resources

        work_dirname = self.work_dirname
        if not path(work_dirname).isdir():
            self.make_dir(work_dirname, self.clobber)

        self.run(*args, **kwargs)

def parse_options(args):
    from optplus import OptionParser, OptionGroup

    usage = "%prog [OPTION]... TASK GENOMEDATA TRAINDIR [IDENTIFYDIR]"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    with OptionGroup(parser, "Data selection") as group:
        group.add_option("-t", "--track", action="append", default=[],
                         metavar="TRACK",
                         help="append TRACK to list of tracks to use, using "
                         " commas to separate tied tracks (default all)")

        group.add_option("--tracks-from", action="load", metavar="FILE",
                         dest="track",
                         help="append tracks from newline-delimited FILE to"
                         " list of tracks to use")

        group.add_option("--file-tracks", action="store_true", default=False,
                         help="Instead of tracks specifing tracks in a single genomedata archive,"
                         "the list of tracks is presumed to specify paths to multiple archives,"
                         "each of which has a track named \"continuous\". The genomedata argument"
                         "is ignored.")

        # This is a 0-based file.
        # I know because ENm008 starts at position 0 in encodeRegions.txt.gz
        group.add_option("--include-coords", metavar="FILE",
                         help="limit to genomic coordinates in FILE"
                         " (default all)")

        # exclude goes after all includes
        group.add_option("--exclude-coords", metavar="FILE",
                         help="filter out genomic coordinates in FILE"
                         " (default none)")

        group.add_option("--resolution", type=int, metavar="RES",
                         help="downsample to every RES bp (default %d)" %
                         RESOLUTION)

        group.add_option("--enforce-coords", action="store_true",
                         help="Use exactly the coordinates in include-coords."
                         " Raises an error if any included positions are not supported by"
                         " the genomedata archive.")

    with OptionGroup(parser, "Model files") as group:
        group.add_option("-i", "--input-master", metavar="FILE",
                         help="use or create input master in FILE"
                         " (default %s)" %
                         make_default_filename(InputMasterSaver.resource_name,
                                               DIRPATH_WORK_DIR_HELP / SUBDIRNAME_PARAMS))

        group.add_option("-s", "--structure", metavar="FILE",
                         help="use or create structure in FILE (default %s)" %
                         make_default_filename(StructureSaver.resource_name))

        group.add_option("-p", "--trainable-params", action="append",
                         default=[], metavar="FILE",
                         help="use or create trainable parameters in FILE"
                         " (default WORKDIR/params/params.params)")

        group.add_option("--dont-train", metavar="FILE",
                         help="use FILE as list of parameters not to train"
                         " (default %s)" %
                         make_default_filename(RES_DONT_TRAIN,
                                               DIRPATH_WORK_DIR_HELP / SUBDIRNAME_AUX))

        group.add_option("--seg-table", metavar="FILE",
                         help="load segment hyperparameters from FILE"
                         " (default none)")

        group.add_option("--semisupervised", metavar="FILE",
                         help="semisupervised segmentation with labels in "
                         "FILE (default none)")

        # XXXmax
        group.add_option("--measure-prop", metavar="FILE",
                         help="run with measure prop graph in FILE (default none)")

        group.add_option("--measure-prop-mu", metavar="FILE", default=1, type=float,
                         help="mu hyperparameter for measure prop")

        group.add_option("--measure-prop-nu", metavar="FILE", default=0, type=float,
                         help="nu hyperparameter for measure prop")

        group.add_option("--measure-prop-weight", metavar="FILE", default=1.0, type=float,
                         help="weight hyperparameter for measure prop")

        group.add_option("--measure-prop-num-iters", metavar="FILE", default=1, type=int,
                         help="number of iterations to run posterior/measure prop")

        group.add_option("--measure-prop-am-num-iters", metavar="FILE", default=100, type=int,
                         help="number of iterations to run alternating minimization in measure prop")

        group.add_option("--measure-prop-reuse-evidence", action="store_true",
                         help="Use evidence from the last round of measure prop to get posteriors for"
                         " the next round. By default, use uniform evidence.")

        # XXXmax
        group.add_option("--virtual-evidence", metavar="FILE",
                         help="supply virtual evidence in FILE (default none)")

        # XXXmax
        group.add_option("--virtual-evidence-dir", metavar="DIR",
                         help="supply virtual evidence in DIR (default none)")

    with OptionGroup(parser, "Intermediate files") as group:
        group.add_option("-o", "--observations", metavar="DIR",
                          help="use or create observations in DIR"
                         " (default %s)" %
                         (DIRPATH_WORK_DIR_HELP / SUBDIRNAME_OBS))

        group.add_option("-r", "--recover", metavar="DIR",
                         help="continue from interrupted run in DIR")

    with OptionGroup(parser, "Output files") as group:
        group.add_option("-b", "--bed", metavar="FILE",
                         help="create identification BED track in FILE"
                         " (default WORKDIR/%s)" % BED_FILEBASENAME)

        group.add_option("--bigBed", metavar="FILE",
                         help="specify layered bigBed filename")

    with OptionGroup(parser, "Modeling variables") as group:
        #group.add_option("-D", "--distribution", choices=DISTRIBUTIONS,
        group.add_option("-D", "--distribution",
                         metavar="DIST",
                         help="use DIST distribution"
                         " (default %s)" % DISTRIBUTION_DEFAULT)

        group.add_option("--num-instances", type=int,
                         default=NUM_INSTANCES, metavar="NUM",
                         help="run NUM training instances, randomizing start"
                         " parameters NUM times (default %d)" % NUM_INSTANCES)

        group.add_option("-N", "--num-labels", type=slice, metavar="SLICE",
                         help="make SLICE segment labels"
                         " (default %d)" % NUM_SEGS)

        group.add_option("--num-sublabels", type=int, metavar="NUM",
                         help="make NUM segment sublabels"
                         " (default %d)" % NUM_SUBSEGS)

        group.add_option("--max-train-rounds", type=int, metavar="NUM",
                         help="each training instance runs a maximum of NUM"
                         " rounds (default %d)" % MAX_EM_ITERS)

        group.add_option("--ruler-scale", type=int, metavar="SCALE",
                         help="ruler marking every SCALE bp (default %d)" %
                         RULER_SCALE)

        group.add_option("--len-prior-strength", type=float, metavar="RATIO",
                         help="use RATIO times the number of data counts as"
                         " the number of pseudocounts for the segment length"
                         " prior (default %f)" % LEN_PRIOR_STRENGTH)

        group.add_option("--graph-prior-strength", type=float, metavar="RATIO",
                         help="use NUM pseudocounts for determining label-label"
                         " transition probabilites.")

        group.add_option("--segtransition-weight-scale", type=float,
                         metavar="SCALE",
                         help="exponent for segment transition probability "
                         " (default %f)" % SEGTRANSITION_WEIGHT_SCALE)

        group.add_option("--reverse-world", action="append", type=int,
                         default=[], metavar="WORLD",
                         help="reverse sequences in concatenated world WORLD"
                         " (0-based)")

    with OptionGroup(parser, "Technical variables") as group:
        group.add_option("-m", "--mem-usage", default=MEM_USAGE_PROGRESSION,
                         metavar="PROGRESSION",
                         help="try each float in PROGRESSION as the number "
                         "of gibibytes of memory to allocate in turn "
                         "(default %s)" % MEM_USAGE_PROGRESSION)

        group.add_option("-S", "--split-sequences", metavar="SIZE",
                         default=MAX_FRAMES, type=int,
                         help="split up sequences that are larger than SIZE "
                         "bp (default %s)" % MAX_FRAMES)

        group.add_option("-v", "--verbosity", type=int, default=VERBOSITY,
                         metavar="NUM",
                         help="show messages with verbosity NUM"
                         " (default %d)" % VERBOSITY)

        group.add_option("--cluster-opt", action="append", default=[],
                         metavar="OPT",
                         help="specify an option to be passed to the "
                         "cluster manager")

    with OptionGroup(parser, "Flags") as group:
        group.add_option("-c", "--clobber", action="store_true",
                         help="delete any preexisting files")
        group.add_option("-n", "--dry-run", action="store_true",
                         help="write all files, but do not run any"
                         " executables")

    options, args = parser.parse_args(args)

    if len(args) < 3:
        parser.error("Expected at least 3 arguments.")
    if args[0] == "train":
        if len(args) != 3:
            parser.error("Expected 3 arguments for the train task.")
    else:
        if len(args) != 4:
            parser.error("Expected 4 arguments for the identify task.")

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    runner = Runner.fromoptions(args, options)

    return runner()

if __name__ == "__main__":
    sys.exit(main())
