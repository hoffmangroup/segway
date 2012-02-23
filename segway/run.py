#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: main Segway implementation
"""

__version__ = "$Revision$"

# Copyright 2008-2012 Michael M. Hoffman <mmh1@uw.edu>

from cStringIO import StringIO
from collections import defaultdict
from contextlib import closing
from copy import copy
from datetime import datetime
from distutils.spawn import find_executable
from errno import EEXIST, ENOENT
from functools import partial
from itertools import count, izip, repeat
from math import ceil, floor, ldexp, log10
from os import environ, extsep
import re
from shutil import copy2
from string import letters
import sys
from tempfile import gettempdir
from threading import Event, Lock, Thread
from time import sleep
from urllib import quote
from uuid import uuid1
from warnings import warn

from genomedata import Genome
from numpy import (append, arange, arcsinh, array, empty, finfo, float32, intc,
                   int64, NINF, square, vstack, zeros)
from optplus import str2slice_or_int
from optbuild import AddableMixin
from path import path
from pkg_resources import Requirement, working_set
from tabdelim import DictReader, ListWriter

from .bed import read_native
from .cluster import (make_native_spec, JobTemplateFactory, RestartableJob,
                      RestartableJobDict, Session)
from .input_master import InputMasterSaver
from .layer import layer, make_layer_filename
from .structure import StructureSaver
from ._util import (ceildiv, data_filename,
                    DTYPE_OBS_INT, DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                    DISTRIBUTION_ASINH_NORMAL, EXT_BED, EXT_FLOAT, EXT_GZ,
                    EXT_INT, EXT_PARAMS, EXT_TAB,
                    extjoin, extjoin_not_none, GB, get_chrom_coords,
                    is_empty_array, ISLAND_BASE_NA, ISLAND_LST_NA, load_coords,
                    _make_continuous_cells, make_default_filename,
                    make_filelistpath, maybe_gzip_open,
                    MB, memoized_property, OFFSET_START, OFFSET_END,
                    OFFSET_STEP, OptionBuilder_GMTK, PassThroughDict, PKG,
                    POSTERIOR_PROG, PREFIX_LIKELIHOOD, PREFIX_PARAMS,
                    _save_observations_window, save_template,
                    SEG_TABLE_WIDTH, SUBDIRNAME_LOG, SUBDIRNAME_PARAMS,
                    SUPERVISION_UNSUPERVISED, SUPERVISION_SEMISUPERVISED,
                    USE_MFSDG, VITERBI_PROG)

# set once per file run
UUID = uuid1().hex

# XXX: I should really get some sort of Enum for this, I think Peter
# Norvig has one
DISTRIBUTIONS = [DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                 DISTRIBUTION_ASINH_NORMAL]
DISTRIBUTION_DEFAULT = DISTRIBUTION_ASINH_NORMAL

MIN_NUM_SEGS = 2
NUM_SEGS = MIN_NUM_SEGS
NUM_SUBSEGS = 1
RULER_SCALE = 10
MAX_EM_ITERS = 100
TEMPDIR_PREFIX = PKG + "-"

MAX_WINDOWS = 1000

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
PRIOR_STRENGTH = 0

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

LOG_LIKELIHOOD_DIFF_FRAC = 1e-5

NUM_SEQ_COLS = 2 # dinucleotide, presence_dinucleotide

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2

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

def make_prefix_fmt(num):
    # make sure there are sufficient leading zeros
    return "%%0%dd." % (int(floor(log10(num))) + 1)

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

BED_FILEBASENAME = extjoin(PKG, EXT_BED, EXT_GZ) # "segway.bed.gz"
BEDGRAPH_FILEBASENAME = extjoin(PREFIX_POSTERIOR, EXT_BEDGRAPH, EXT_GZ) # "posterior%s.bed.gz"
FLOAT_TABFILEBASENAME = extjoin("observations", EXT_TAB)
TRAIN_FILEBASENAME = extjoin(PREFIX_TRAIN, EXT_TAB)

SUBDIRNAME_ACC = "accumulators"
SUBDIRNAME_AUX = "auxiliary"
SUBDIRNAME_LIKELIHOOD = "likelihood"
SUBDIRNAME_OBS = "observations"
SUBDIRNAME_POSTERIOR = "posterior"
SUBDIRNAME_VITERBI = "viterbi"

SUBDIRNAMES_EITHER = [SUBDIRNAME_AUX]
SUBDIRNAMES_TRAIN = [SUBDIRNAME_ACC, SUBDIRNAME_LIKELIHOOD,
                     SUBDIRNAME_PARAMS]

FLOAT_TAB_FIELDNAMES = ["filename", "window_index", "chrom", "start", "end"]
JOB_LOG_FIELDNAMES = ["jobid", "jobname", "prog", "num_segs",
                      "num_frames", "maxvmem", "cpu", "exit_status"]
# XXX: should add num_subsegs as well, but it's complicated to pass
# that data into RestartableJobDict.wait()

TRAIN_FIELDNAMES = ["name", "value"]

TRAIN_OPTION_TYPES = \
    dict(input_master_filename=str, structure_filename=str,
         params_filename=str, dont_train_filename=str, seg_table_filename=str,
         distribution=str, len_seg_strength=float,
         segtransition_weight_scale=float, ruler_scale=int, resolution=int,
         num_segs=int, num_subsegs=int, track_specs=[str])

# templates and formats
RES_OUTPUT_MASTER = "output.master"
RES_DONT_TRAIN = "dont_train.list"
RES_INC_TMPL = "segway.inc.tmpl"
RES_SEG_TABLE = "seg_table.tab"

TRACK_FMT = "browser position %s:%s-%s"
FIXEDSTEP_FMT = "fixedStep chrom=%s start=%s step=1 span=1"

BED_ATTRS = dict(autoScale="off")
BED_ATTRS_VITERBI = dict(name="%s.%s" % (PKG, UUID),
                         visibility="dense",
                         viewLimits="0:1",
                         itemRgb="on",
                         **BED_ATTRS)

BED_DESC_VITERBI = "%s segmentation of %%s" % PKG

TRAIN_ATTRNAMES = ["input_master_filename", "params_filename",
                   "log_likelihood_filename"]
LEN_TRAIN_ATTRNAMES = len(TRAIN_ATTRNAMES)

COMMENT_POSTERIOR_TRIANGULATION = \
    "%% triangulation modified for posterior decoding by %s" % PKG

FIXEDSTEP_HEADER = "fixedStep chrom=%s start=%d step=%d span=%d"
POSTERIOR_BEDGRAPH_HEADER="track type=bedGraph name=posterior.%d \
        description=\"Segway posterior probability of label %d\" \
        visibility=dense  viewLimits=0:100 maxHeightPixels=0:0:10 \
        autoScale=off color=200,100,0 altColor=0,100,200"

# training results
# XXX: Python 2.6: this should really be a namedtuple, yuck
OFFSET_NUM_SEGS = 1
OFFSET_FILENAMES = 2 # where the filenames begin in the results
OFFSET_PARAMS_FILENAME = 3

SUPERVISION_LABEL_OFFSET = 1

RESOLUTION = 1

INDEX_BED_START = 1

SEGTRANSITION_WEIGHT_SCALE = 1.0

DIRPATH_WORK_DIR_HELP = path("WORKDIR")

# 62 so that it's not in sync with the 10 second job wait sleep time
THREAD_START_SLEEP_TIME = 62 # XXX: this should become an option

## functions
try:
    # XXX: new in version 2.6? 2.7?
    from itertools import permutations
except ImportError:
    # copied from
    # http://docs.python.org/library/itertools.html#itertools.permutations
    def permutations(iterable, r=None):
        # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        # permutations(range(3)) --> 012 021 102 120 201 210
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = range(n)
        cycles = range(n, n-r, -1)
        yield tuple(pool[i] for i in indices[:r])
        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return

def quote_trackname(text):
    # legal characters are ident in GMTK_FileTokenizer.ll:
    # {alpha})({alpha}|{dig}|\_|\-)* (alpha is [A-za-z], dig is [0-9])
    res = text.replace("_", "_5f")
    res = res.replace(".", "_2e")

    # quote eliminates everything except for _.-, replaces with % escapes
    res = quote(res)
    res = res.replace("%", "_")

    # add stub to deal with non-alphabetic first characters
    if res[0] not in letters:
        # __ should never appear in strings quoted as before
        res = "x__" + res

    return res

def make_fixedstep_header(chrom, start, resolution):
    """
    this function expects 0-based coordinates
    it does the conversion to 1-based coordinates for you
    """
    start_1based = start+1

    # XXX: if there is an overhang of less than resolution, then
    # having step/span = resolution means the last datum in a window
    # will actually extend too far. There's no point in fixing this
    # now, since we want to switch to bedGraph eventually anyway
    return FIXEDSTEP_HEADER % (chrom, start_1based, resolution, resolution)

def make_bed_attr(key, value):
    if " " in value:
        value = '"%s"' % value

    return "%s=%s" % (key, value)

def make_bed_attrs(mapping):
    res = " ".join(make_bed_attr(key, value)
                   for key, value in mapping.iteritems())

    return "track %s" % res

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

def parse_bed4(line):
    row = line.split()
    chrom, start, end, seg = row[:4]
    return row, (chrom, start, end, seg)

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

def convert_windows(attrs, name):
    supercontig_start = attrs.start
    edges_array = getattr(attrs, name) + supercontig_start

    return edges_array.tolist()

def is_training_progressing(last_ll, curr_ll,
                            min_ll_diff_frac=LOG_LIKELIHOOD_DIFF_FRAC):
    # using x !< y instead of x >= y to give the right default answer
    # in the case of NaNs
    return not abs((curr_ll - last_ll)/last_ll) < min_ll_diff_frac

class NoData(object):
    """
    sentinel for not adding an extra field to coords, so that one can
    still use None
    """

def find_overlaps_include(start, end, coords, data=repeat(NoData)):
    """
    find items in coords that overlap (start, end)

    NOTE: multiple overlapping regions in coords will result in data
    being considered more than once
    """
    # diagram of how this works:
    #        --------------------- (include)
    #
    # various (start, end) cases:
    # A   --
    # B                              --
    # C   --------------------------
    # D   ------
    # E                       -------------
    # F             ---------
    res = []

    for (include_start, include_end), datum in izip(coords, data):
        if start > include_end or end <= include_start:
            # cases A, B
            continue
        elif start <= include_start:
            if end < include_end:
                # case D
                include_end = end

            # case C otherwise
        elif start > include_start:
            include_start = start

            if end < include_end:
                # case F
                include_end = end

            # case E otherwise
        else:
            assert False # can't happen

        item = [include_start, include_end]

        if datum is not NoData:
            item.append(datum)

        res.append(item)

    return res

def find_overlaps_exclude(include_coords, exclude_coords):
    # diagram of how this works:
    #        --------------------- (include)
    #
    # various exclude cases:
    # A   --
    # B                              --
    # C   --------------------------
    # D   ------
    # E                       -------------
    # F             ---------

    # does nothing if exclude_coords is empty
    for exclude_start, exclude_end in exclude_coords:
        new_include_coords = []

        for include_index, include_coord in enumerate(include_coords):
            include_start, include_end = include_coord

            if exclude_start > include_end or exclude_end <= include_start:
                # cases A, B
                new_include_coords.append([include_start, include_end])
            elif exclude_start <= include_start:
                if exclude_end >= include_end:
                    # case C
                    pass
                else:
                    # case D
                    new_include_coords.append([exclude_end, include_end])
            elif exclude_start > include_start:
                if exclude_end >= include_end:
                    # case E
                    new_include_coords.append([include_start, exclude_start])
                else:
                    # case F
                    new_include_coords.append([include_start, exclude_start])
                    new_include_coords.append([exclude_end, include_end])

            else:
                assert False # can't happen

        include_coords = new_include_coords

    return include_coords

def find_overlaps(start, end, include_coords, exclude_coords):
    if include_coords is None or is_empty_array(include_coords):
        res = [[start, end]]
    else:
        res = find_overlaps_include(start, end, include_coords)

    if exclude_coords is None or is_empty_array(exclude_coords):
        return res
    else:
        return find_overlaps_exclude(res, exclude_coords)

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

def update_starts(starts, ends, new_starts, new_ends, instance_index):
    next_index = instance_index + 1

    starts[next_index:next_index] = new_starts
    ends[next_index:next_index] = new_ends

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

        self.posterior_clique_indices = POSTERIOR_CLIQUE_INDICES.copy()

        self.triangulation_filename_is_new = None

        self.supervision_coords = None
        self.supervision_labels = None

        self.card_supervision_label = -1

        self.include_tracknames = []
        self.tied_tracknames = {} # dict of head trackname -> tied tracknames

        # default is 0
        self.global_mem_usage = LockableDefaultDict(int)

        # data
        # a "window" is what GMTK calls a segment
        self.num_windows = None
        self.window_coords = None
        self.mins = None
        self.maxs = None
        self.tracknames = None # encoded/quoted version

        # variables
        self.num_segs = NUM_SEGS
        self.num_subsegs = NUM_SUBSEGS
        self.num_instances = NUM_INSTANCES
        self.len_seg_strength = PRIOR_STRENGTH
        self.distribution = DISTRIBUTION_DEFAULT
        self.max_em_iters = MAX_EM_ITERS
        self.max_frames = MAX_FRAMES
        self.segtransition_weight_scale = SEGTRANSITION_WEIGHT_SCALE
        self.ruler_scale = RULER_SCALE
        self.resolution = RESOLUTION

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
        if value is not None:
            setattr(self, name, value)

    options_to_attrs = [("recover", "recover_dirname"),
                        ("observations", "obs_dirname"),
                        ("bed", "bed_filename"),
                        ("semisupervised", "supervision_filename"),
                        ("bigBed", "bigbed_filename"),
                        ("include_coords", "include_coords_filename"),
                        ("exclude_coords", "exclude_coords_filename"),
                        ("num_instances",),
                        ("verbosity",),
                        ("split_sequences", "max_frames"),
                        ("clobber",),
                        ("dry_run",),
                        ("input_master", "input_master_filename"),
                        ("structure", "structure_filename"),
                        ("dont_train", "dont_train_filename"),
                        ("seg_table", "seg_table_filename"),
                        ("distribution",),
                        ("prior_strength", "len_seg_strength"),
                        ("segtransition_weight_scale",),
                        ("ruler_scale",),
                        ("resolution",),
                        ("num_labels", "num_segs"),
                        ("num_sublabels", "num_subsegs"),
                        ("max_train_rounds", "max_em_iters"),
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
        tied_tracknames = defaultdict(list)
        head_tracknames = {}
        head_trackname_list = []
        used_tracknames = set()

        for track_spec in res.track_specs:
            current_tracknames = track_spec.split(",")

            if not used_tracknames.isdisjoint(current_tracknames):
                raise ValueError("can't tie one track in multiple groups")

            include_tracknames.extend(current_tracknames)
            used_tracknames |= frozenset(current_tracknames)

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
        return load_coords(self.include_coords_filename)

    @memoized_property
    def exclude_coords(self):
        return load_coords(self.exclude_coords_filename)

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
                assert len_slice.step == ruler_scale

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

    @memoized_property
    def viterbi_filenames(self):
        self.make_subdir(SUBDIRNAME_VITERBI)
        return self._make_viterbi_filenames(self.work_dirpath)

    @memoized_property
    def posterior_filenames(self):
        self.make_subdir(SUBDIRNAME_POSTERIOR)
        return map(self.make_posterior_filename, xrange(self.num_windows))

    @memoized_property
    def recover_viterbi_filenames(self):
        recover_dirpath = self.recover_dirpath
        if recover_dirpath:
            return self._make_viterbi_filenames(recover_dirpath)
        else:
            return None

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
        return [end - start for chr, start, end in self.window_coords]

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
        return self.work_dirpath / BED_FILEBASENAME

    @memoized_property
    def bedgraph_filename(self):
        return self.work_dirpath / BEDGRAPH_FILEBASENAME

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
        return len(self.track_indexes)

    @memoized_property
    def track_indexes_text(self):
        return ",".join(map(str, self.track_indexes))

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

    def transform(self, num):
        if self.distribution == DISTRIBUTION_ASINH_NORMAL:
            return arcsinh(num)
        else:
            return num

    def make_cpp_options(self, input_params_filename=None,
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
            track_indexes = array([], intc)
            tied_track_indexes_list = []
            self.float_filelistpath = None # no float data

        else:
            # default: use all tracks in archive
            track_indexes = arange(len(tracknames))

            assert not self.tied_tracknames
            tied_track_indexes_list = [[track_index]
                                       for track_index in track_indexes]
            unquoted_tracknames = tracknames

        # replace illegal characters in tracknames only, not unquoted_tracknames
        tracknames = map(quote_trackname, tracknames)

        # assert: none of the quoted tracknames are the same
        assert len(tracknames) == len(frozenset(tracknames))

        self.tracknames = tracknames
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

    def make_obs_filepath(self, dirpath, prefix, suffix):
        return dirpath / (prefix + suffix)

    def make_obs_filepaths(self, chrom, window_index, temp=False):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_WINDOWS))
        prefix = prefix_feature_tmpl % window_index

        if temp:
            prefix = "".join([prefix, UUID, extsep])
            dirpath = path(gettempdir())
        else:
            dirpath = self.obs_dirpath

        make_obs_filepath_custom = partial(self.make_obs_filepath, dirpath,
                                           prefix)

        return (make_obs_filepath_custom(EXT_FLOAT),
                make_obs_filepath_custom(EXT_INT))

    def print_obs_filepaths(self, float_filelist, int_filelist,
                            *args, **kwargs):
        float_filepath, int_filepath = self.make_obs_filepaths(*args, **kwargs)
        print >>float_filelist, float_filepath
        print >>int_filelist, int_filepath

        return float_filepath, int_filepath

    def save_resource(self, resname, subdirname=""):
        orig_filename = data_filename(resname)

        orig_filepath = path(orig_filename)
        dirpath = self.work_dirpath / subdirname

        orig_filepath.copy(dirpath)
        return dirpath / orig_filepath.name

    def save_include(self):
        num_segs = self.num_segs

        if isinstance(num_segs, slice):
            num_segs = "undefined\n#error must define CARD_SEG"

        resolution = self.resolution
        ruler_scale = self.ruler_scale
        if ruler_scale % resolution != 0:
            msg = ("resolution %d is not a divisor of ruler scale %d"
                   % (resolution, ruler_scale))
            raise ValueError(msg)
        ruler_scale_scaled = ruler_scale // resolution

        mapping = dict(card_seg=num_segs,
                       card_subseg=self.num_subsegs,
                       card_presence=resolution+1,
                       card_segCountDown=self.card_seg_countdown,
                       card_supervisionLabel=self.card_supervision_label,
                       card_frameIndex=self.max_frames,
                       ruler_scale=ruler_scale_scaled)

        aux_dirpath = self.work_dirpath / SUBDIRNAME_AUX

        self.gmtk_include_filename, self.gmtk_include_filename_is_new = \
            save_template(self.gmtk_include_filename, RES_INC_TMPL, mapping,
                          aux_dirpath, self.clobber)

    def save_observations_window(self, float_filename, int_filename, float_data,
                                seq_data=None, supervision_data=None):
        return _save_observations_window(float_filename, int_filename,
                                         float_data, self.resolution,
                                         self.distribution, seq_data,
                                         supervision_data)

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

    def write_observations(self, float_filelist, int_filelist, float_tabfile):
        # XXX: these expect different filepaths
        assert not ((self.identify or self.posterior) and self.train)

        print_obs_filepaths_custom = partial(self.print_obs_filepaths,
                                             float_filelist, int_filelist,
                                             temp=self.identify)
        save_observations_window = self.save_observations_window

        float_tabwriter = ListWriter(float_tabfile)
        float_tabwriter.writerow(FLOAT_TAB_FIELDNAMES)

        zipper = izip(count(), self.used_supercontigs, self.window_coords)

        for window_index, supercontig, (chrom, start, end) in zipper:
            float_filepath, int_filepath = \
                print_obs_filepaths_custom(chrom, window_index)

            row = [float_filepath, str(window_index), chrom, str(start),
                   str(end)]
            float_tabwriter.writerow(row)
            print >>sys.stderr, " %s (%d, %d)" % (float_filepath, start, end)

            # if they don't both exist
            if not (float_filepath.exists() and int_filepath.exists()):
                # XXX: next several lines are duplicative
                seq_cells = self.make_seq_cells(supercontig, start, end)

                supervision_cells = \
                    self.make_supervision_cells(chrom, start, end)

                if __debug__:
                    if self.identify:
                        assert seq_cells is None and supervision_cells is None

                if not self.train:
                    # don't actually write data
                    continue

                continuous_cells = \
                    self.make_continuous_cells(supercontig, start, end)

                save_observations_window(float_filepath, int_filepath,
                                        continuous_cells, seq_cells,
                                        supervision_cells)

    def make_continuous_cells(self, supercontig, start, end):
        return _make_continuous_cells(supercontig, start, end,
                                      self.track_indexes)

    def make_supervision_cells(self, chrom, start, end):
        supervision_type = self.supervision_type
        if supervision_type == SUPERVISION_UNSUPERVISED:
            return

        assert supervision_type == SUPERVISION_SEMISUPERVISED

        coords_chrom = self.supervision_coords[chrom]
        labels_chrom = self.supervision_labels[chrom]

        res = zeros(end - start, dtype=DTYPE_OBS_INT)

        supercontig_coords_labels = find_overlaps_include(start, end,
                                                          coords_chrom,
                                                          labels_chrom)

        for label_start, label_end, label_index in supercontig_coords_labels:
            # adjust so that zero means no label
            label_adjusted = label_index + SUPERVISION_LABEL_OFFSET
            res[label_start-start:label_end-start] = label_adjusted

        return res

    def make_seq_cells(self, supercontig, start, end):
        if not self.use_dinucleotide:
            return

        seq = supercontig.seq
        len_seq = len(seq)

        # window_start: relative to the beginning of the
        # supercontig
        window_start = start - supercontig.start
        window_end = end - supercontig.start

        if window_end < len_seq:
            return seq[window_start:window_end+1]
        elif window_end == len(seq):
            seq_window = seq[window_start:window_end]
            return append(seq_window, ord("N"))
        else:
            raise ValueError("sequence too short for supercontig")

    def prep_observations(self, genome):
        # XXX: this function is way too long, try to get it to fit
        # inside a screen on your enormous monitor
        include_coords = self.include_coords
        exclude_coords = self.exclude_coords

        window_index = 0
        window_coords = []
        num_bases = 0
        used_supercontigs = [] # continuous objects
        max_frames = self.max_frames
        num_tracks = self.num_tracks

        # XXX: use groupby(include_coords) and then access chromosomes
        # randomly rather than iterating through them all

        # XXX: hopefully this will fix a bug where end of one = start
        # of next results in a MemoryError crash (email from ML to
        # MMH, 2012/1/24)
        for chromosome in genome:
            chrom = chromosome.name

            chr_include_coords = get_chrom_coords(include_coords, chrom)
            chr_exclude_coords = get_chrom_coords(exclude_coords, chrom)

            if (chr_include_coords is not None
                and is_empty_array(chr_include_coords)):
                continue

            for supercontig, continuous in chromosome.itercontinuous():
                assert continuous is None or continuous.shape[1] >= num_tracks

                if continuous is None:
                    starts = [supercontig.start]
                    ends = [supercontig.end]
                else:
                    attrs = supercontig.attrs
                    starts = convert_windows(attrs, "chunk_starts")
                    ends = convert_windows(attrs, "chunk_ends")

                ## iterate through windows and write
                ## izip so it can be modified in place
                for instance_index, start, end in izip(count(), starts, ends):
                    if include_coords or exclude_coords:
                        overlaps = find_overlaps(start, end,
                                                 chr_include_coords,
                                                 chr_exclude_coords)
                        len_overlaps = len(overlaps)

                        if len_overlaps == 0:
                            continue
                        elif len_overlaps == 1:
                            start, end = overlaps[0]
                        else:
                            new_starts, new_ends = zip(*overlaps)
                            update_starts(starts, ends, new_starts, new_ends,
                                          instance_index)
                            continue # consider the newly split sequences next

                    num_bases_window = end - start
                    num_frames = ceildiv(num_bases_window, self.resolution)
                    if not MIN_FRAMES <= num_frames:
                        text = " skipping short sequence of length %d" % num_frames
                        print >>sys.stderr, text
                        continue

                    if num_frames > max_frames:
                        # XXX: I really ought to check that this is
                        # going to always work even for corner cases,
                        # but if it doesn't I am saved by another
                        # split later on

                        # split_sequences was True, so split them
                        num_new_starts = ceildiv(num_frames, max_frames)

                        # // means floor division
                        offset = (num_frames // num_new_starts)
                        new_offsets = arange(num_new_starts) * offset
                        new_starts = start + new_offsets
                        new_ends = append(new_starts[1:], end)

                        update_starts(starts, ends, new_starts, new_ends,
                                      instance_index)
                        continue

                    # start: relative to beginning of chromosome
                    window_coords.append((chrom, start, end))
                    used_supercontigs.append(supercontig)

                    num_bases += num_bases_window

                    window_index += 1

        self.subset_metadata(genome)

        self.num_windows = window_index # already has +1 added to it
        self.num_bases = num_bases
        self.window_coords = window_coords

        self.used_supercontigs = used_supercontigs

    def open_writable_or_dummy(self, filepath):
        if not filepath or (not self.clobber and filepath.exists()):
            return closing(StringIO()) # dummy output
        else:
            return open(filepath, "w")

    def save_observations(self):
        open_writable = partial(self.open_writable_or_dummy)

        with open_writable(self.float_filelistpath) as float_filelist:
            with open_writable(self.int_filelistpath) as int_filelist:
                with open_writable(self.float_tabfilepath) as float_tabfile:
                    self.write_observations(float_filelist, int_filelist,
                                            float_tabfile)

    def save_input_master(self, instance_index=None, new=False):
        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        self.input_master_filename, input_master_filename_is_new = \
            InputMasterSaver(self)(input_master_filename, self.params_dirpath,
                                   self.clobber, instance_index)

    def _make_viterbi_filenames(self, dirpath):
        viterbi_dirpath = dirpath / SUBDIRNAME_VITERBI
        num_windows = self.num_windows

        viterbi_filename_fmt = (PREFIX_VITERBI + make_prefix_fmt(num_windows)
                                + EXT_BED)
        return [viterbi_dirpath / viterbi_filename_fmt % index
                for index in xrange(num_windows)]

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

    def save_structure(self):
        self.structure_filename, _ = \
            StructureSaver(self)(self.structure_filename, self.work_dirname,
                                 self.clobber)

    def save_observations_params(self):
        self.load_supervision()

        # need to open Genomedata archive first in order to determine
        # self.tracknames and self.num_tracks
        with Genome(self.genomedataname) as genome:
            self.set_tracknames(genome)
            self.prep_observations(genome)
            self.save_observations()
            self.used_supercontigs = None # no longer necessary

        if self.train:
            self.set_log_likelihood_filenames()

        self.save_include()
        self.set_params_filename()
        self.save_structure()

    def copy_results(self, name, src_filename, dst_filename):
        if dst_filename:
            copy2(src_filename, dst_filename)
        else:
            dst_filename = src_filename

        setattr(self, name, dst_filename)

    def make_bed_desc_attrs(self, mapping, desc_tmpl):
        attrs = mapping.copy()
        attrs["description"] = desc_tmpl % ", ".join(self.unquoted_tracknames)

        return make_bed_attrs(attrs)

    def make_bed_header_viterbi(self):
        return self.make_bed_desc_attrs(BED_ATTRS_VITERBI, BED_DESC_VITERBI)

    def concatenate_bed(self):
        # the final bed filename, not the individual viterbi_filenames
        bed_filename = self.bed_filename

        # values for comparison to combine adjoining segments
        last_line = ""
        last_start = None
        last_vals = (None, None, None) # (chrom, coord, seg)

        with maybe_gzip_open(bed_filename, "w") as bed_file:
            # XXX: add in browser track line (see SVN revisions
            # previous to 195)
            print >>bed_file, self.make_bed_header_viterbi()

            for viterbi_filename in self.viterbi_filenames:
                with open(viterbi_filename) as viterbi_file:
                    lines = viterbi_file.readlines()
                    first_line = lines[0]
                    first_row, first_coords = parse_bed4(first_line)
                    (chrom, start, end, seg) = first_coords

                    # write the last line and the first line, after
                    # potentially merging
                    if last_vals == (chrom, start, seg):
                        first_row[INDEX_BED_START] = last_start

                        # add back trailing newline eliminated by line.split()
                        merged_line = "\t".join(first_row) + "\n"

                        # if there's just a single line in the BED file
                        if len(lines) == 1:
                            last_line = merged_line
                            last_vals = (chrom, end, seg)
                            # last_start is already set correctly
                            # postpone writing until after additional merges
                            continue
                        else:
                            # write the merged line
                            bed_file.write(merged_line)
                    else:
                        if len(lines) == 1:
                            # write the last line of the last file.
                            # hold back the first line of this file,
                            # and treat it as the last line
                            bed_file.write(last_line)
                        else:
                            # write the last line of the last file, first
                            # line of this file
                            bed_file.writelines([last_line, first_line])

                    # write the bulk of the lines
                    bed_file.writelines(lines[1:-1])

                    # set last_line
                    last_line = lines[-1]
                    last_row, last_coords = parse_bed4(last_line)
                    (chrom, start, end, seg) = last_coords
                    last_vals = (chrom, end, seg)
                    last_start = start

            # write the very last line of all files
            bed_file.write(last_line)


    def concatenate_bedgraph(self):
        # the final bedgraph filename, not the individual posterior_filenames
        bedgraph_filename = self.bedgraph_filename

        for num_seg in xrange(self.num_segs):
            # values for comparison to combine adjoining segments
            last_start = None
            last_line = ""
            last_vals = (None, None, None) # (chrom, coord, seg)

            with maybe_gzip_open(bedgraph_filename % num_seg, "w") as bedgraph_file:
                # XXX: add in browser track line (see SVN revisions
                # previous to 195)
                posterior_header = POSTERIOR_BEDGRAPH_HEADER % (num_seg,
                                                        num_seg)
                print >>bedgraph_file, posterior_header

                for posterior_filename in self.posterior_filenames:
                    with open(posterior_filename % num_seg) as posterior_file:
                        lines = posterior_file.readlines()
                        first_line = lines[0]
                        first_row, first_coords = parse_bed4(first_line)
                        (chrom, start, end, seg) = first_coords

                        # write the last line and the first line, after
                        # potentially merging
                        if last_vals == (chrom, start, seg):
                            first_row[INDEX_BED_START] = last_start

                            # add back trailing newline eliminated by line.split()
                            merged_line = "\t".join(first_row) + "\n"

                            # if there's just a single line in the BED file
                            if len(lines) == 1:
                                last_line = merged_line
                                last_vals = (chrom, end, seg)
                                # last_start is already set correctly
                                # postpone writing until after additional merges
                                continue
                            else:
                                # write the merged line
                                bedgraph_file.write(merged_line)
                        else:
                            if len(lines) == 1:
                                # write the last line of the last file.
                                # hold back the first line of this file,
                                # and treat it as the last line
                                bedgraph_file.write(last_line)
                            else:
                                # write the last line of the last file, first
                                # line of this file
                                bedgraph_file.writelines([last_line, first_line])

                        # write the bulk of the lines
                        bedgraph_file.writelines(lines[1:-1])

                        # set last_line
                        last_line = lines[-1]
                        last_row, last_coords = parse_bed4(last_line)
                        (chrom, start, end, seg) = last_coords
                        last_vals = (chrom, end, seg)
                        last_start = start

                # write the very last line of all files
                bedgraph_file.write(last_line)

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
        return self.make_filename(PREFIX_POSTERIOR, window_index, EXT_BED,
                                  subdirname=SUBDIRNAME_POSTERIOR)

    def make_job_name_train(self, instance_index, round_index, window_index):
        return "%s%d.%d.%s.%s.%s" % (PREFIX_JOB_NAME_TRAIN, instance_index,
                                     round_index, window_index,
                                     self.work_dirpath.name, UUID)

    def make_job_name_identify(self, prefix, window_index):
        return "%s%d.%s.%s" % (prefix, window_index, self.work_dirpath.name,
                               UUID)

    def make_gmtk_kwargs(self):
        """
        universal args to gmtkEMtrain, gmtkViterbi, gmtkJT
        """
        res = dict(strFile=self.structure_filename,
                   verbosity=self.verbosity,
                   island=ISLAND,
                   componentCache=COMPONENT_CACHE,
                   deterministicChildrenStore=DETERMINISTIC_CHILDREN_STORE,
                   jtFile=self.jt_info_filename,
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

        if self.dry_run:
            return None

        session = self.session
        job_tmpl = session.createJobTemplate()

        job_tmpl.jobName = job_name
        job_tmpl.remoteCommand = ENV_CMD
        job_tmpl.args = map(str, args)

        # this is going to cause problems on heterogeneous systems
        job_tmpl.jobEnvironment = environ

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
        kwargs["cppCommandOptions"] = self.make_cpp_options(input_params_filename)

        res = RestartableJobDict(self.session, self.job_log_file)

        make_acc_filename_custom = partial(self.make_acc_filename, instance_index)

        for window_index, window_len in self.window_lens_sorted():
            acc_filename = make_acc_filename_custom(window_index)
            kwargs_window = dict(trrng=window_index, storeAccFile=acc_filename,
                                **kwargs)

            # -dirichletPriors T only on the first window
            kwargs_window["dirichletPriors"] = (window_index == 0)

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

        cpp_options = self.make_cpp_options(input_params_filename,
                                            output_params_filename)

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

        # print >>sys.stderr, ("setting triangulation_filename = %s"
        #                     % self.triangulation_filename)

    def run_triangulate_single(self, num_segs, num_subsegs=None):
        # print >>sys.stderr, "running triangulation"
        prog = self.prog_factory(TRIANGULATE_PROG)

        self.set_triangulation_filename(num_segs, num_subsegs)

        cpp_options = self.make_cpp_options()
        kwargs = dict(strFile=self.structure_filename,
                      cppCommandOptions=cpp_options,
                      outputTriangulatedFile=self.triangulation_filename,
                      verbosity=self.verbosity)

        # XXX: need exist/clobber logic here
        # XXX: repetitive with queue_gmtk
        self.log_cmdline(prog.build_cmdline(options=kwargs))

        prog(**kwargs)

    def run_triangulate(self):
        num_segs_range = slice2range(self.num_segs)
        for num_segs in num_segs_range:
            self.run_triangulate_single(num_segs)

    def run_train_round(self, instance_index, round_index, **kwargs):
        """
        returns None: normal
        returns not None: abort
        """
        last_params_filename = self.last_params_filename
        curr_params_filename = extjoin(self.params_filename, str(round_index))

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
        # make new files if there is more than one instance
        self.set_triangulation_filename()

        new = self.instance_make_new_params

        instance_index = self.instance_index
        self.set_log_likelihood_filenames(instance_index, new)
        self.set_params_filename(instance_index, new)

        last_log_likelihood, log_likelihood, round_index = \
            self.recover_train_instance()

        if round_index == 0:
            # if round > 0, this is set by self.recover_train_instance()
            self.save_input_master(instance_index, new)

        kwargs = dict(objsNotToTrain=self.dont_train_filename,
                      maxEmIters=1,
                      lldp=LOG_LIKELIHOOD_DIFF_FRAC*100.0,
                      triFile=self.triangulation_filename,
                      **self.make_gmtk_kwargs())

        while (round_index < self.max_em_iters and
               is_training_progressing(last_log_likelihood, log_likelihood)):
            self.run_train_round(instance_index, round_index, **kwargs)

            if self.dry_run:
                return (None, None, None, None)

            last_log_likelihood = log_likelihood
            log_likelihood = self.load_log_likelihood()

            # print >>sys.stderr, "log likelihood = %s" % log_likelihood
            # print >>sys.stderr, "info criterion = %s" % info_criterion

            round_index += 1

        # log_likelihood, num_segs and a list of src_filenames to save
        return (log_likelihood, self.num_segs, self.input_master_filename,
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

    def run_train(self):
        num_instances = self.num_instances
        assert num_instances >= 1

        # save the destination file for input_master as we will be
        # generating new input masters for each start

        # must be before file creation. Otherwise
        # input_master_filename_is_new will be wrong
        input_master_filename, input_master_filename_is_new = \
            InputMasterSaver(self)(self.input_master_filename,
                                   self.params_dirpath, self.clobber)

        self.input_master_filename = input_master_filename

        # should I make new parameters in each instance?
        instance_make_new_params = (num_instances > 1
                                    or isinstance(self.num_segs, slice))
        self.instance_make_new_params = instance_make_new_params
        if not instance_make_new_params:
            self.save_input_master()

        ## save file locations to tab-delimited file
        self.save_train_options()

        if not input_master_filename_is_new:
            # do not overwrite existing file
            input_master_filename = None

        dst_filenames = [input_master_filename,
                         self.params_filename,
                         self.log_likelihood_filename]

        ## which thread runner should I use?
        num_segs_range = slice2range(self.num_segs)

        if len(num_segs_range) > 1 or num_instances > 1:
            run_train_func = self.run_train_multithread
        else:
            run_train_func = self.run_train_singlethread

        ## this is where the actual training takes place
        instance_params = run_train_func(num_segs_range)

        if self.instance_make_new_params:
            self.proc_train_results(instance_params, dst_filenames)
        elif not self.dry_run:
            # only one instance
            assert len(instance_params) == 1
            last_params_filename = instance_params[0][OFFSET_PARAMS_FILENAME]
            copy2(last_params_filename, self.params_filename)

            # always overwrite params.params
            copy2(last_params_filename, self.make_params_filename())

    def run_train_singlethread(self, num_segs_range):
        # having a single-threaded version makes debugging much easier
        with Session() as session:
            self.session = session
            self.instance_index = 0
            res = [self.run_train_instance()]

        self.session = None

        return res

    def run_train_multithread(self, num_segs_range):
        # XXX: Python 2.6: use itertools.product()
        seg_instance_indexes = xrange(self.num_instances)
        enumerator = enumerate((num_seg, seg_instance_index)
                               for num_seg in num_segs_range
                               for seg_instance_index in seg_instance_indexes)

        # ensure memoization before threading
        self.triangulation_dirpath
        self.jt_info_filename
        self.include_cords
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
        max_params = max(instance_params)

        self.num_segs = max_params[OFFSET_NUM_SEGS]
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
        old_filename = make_default_filename(resource,
                                             self.recover_params_dirpath,
                                             instance_index)

        new_filename = make_default_filename(resource, self.params_dirpath,
                                             instance_index)

        path(old_filename).copy2(new_filename)
        return new_filename

    def recover_train_instance(self):
        """
        returns last_log_likelihood, log_likelihood, round_idnex
        NINF, NINF, 0 if there is no recovery
        """
        last_log_likelihood = NINF
        log_likelihood = NINF
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

        window_coords = self.window_coords[window_index]
        (window_chrom, window_start, window_end) = window_coords

        # XXX: duplicative
        row, line_coords = parse_bed4(lines[0])
        (line_chrom, line_start, line_end, seg) = line_coords
        if line_chrom != window_chrom or int(line_start) != window_start:
            return False

        row, line_coords = parse_bed4(lines[-1])
        (line_chrom, line_start, line_end, seg) = line_coords
        if line_chrom != window_chrom or int(line_end) != window_end:
            return False

        # copy the old filename to where the job's output would have
        # landed
        path(recover_filename).copy2(self.viterbi_filenames[window_index])

        print >>sys.stderr, "window %d already complete" % window_index

        return True

    def _queue_identify(self, restartable_jobs, window_index, prefix_job_name,
                        prog, kwargs, output_filenames):
        prog = self.prog_factory(prog)
        job_name = self.make_job_name_identify(prefix_job_name, window_index)
        output_filename = output_filenames[window_index]

        kwargs = self.get_identify_kwargs(window_index, kwargs)

        if prog == VITERBI_PROG:
            kind = "viterbi"
        else:
            kind = "posterior"
        window_coord = self.window_coords[window_index]
        window_chrom, window_start, window_end = window_coord

        float_filepath, int_filepath = \
            self.make_obs_filepaths(window_chrom, window_index, temp=True)

        prefix_args = [find_executable("segway-task"), "run", kind,
                       output_filename, window_chrom, window_start,
                       window_end, self.resolution, self.num_segs,
                       self.genomedataname, float_filepath,
                       int_filepath, self.distribution,
                       self.track_indexes_text]
        output_filename = None

        num_frames = self.window_lens[window_index]

        restartable_job = self.queue_gmtk(prog, kwargs, job_name,
                                          num_frames,
                                          output_filename=output_filename,
                                          prefix_args=prefix_args)

        restartable_jobs.queue(restartable_job)

    def get_identify_kwargs(self, window_index, extra_kwargs):
        cpp_command_options = self.make_cpp_options(self.params_filename)

        res = dict(inputMasterFile=self.input_master_filename,
                   cppCommandOptions=cpp_command_options,
                   cliqueTableNormalize="0.0",
                   **self.make_gmtk_kwargs())

        res.update(extra_kwargs)

        return res

    def run_identify_posterior(self):
        self.instance_index = "identify"

        ## setup files
        if not self.input_master_filename:
            warn("Input master not specified. Generating.")
            self.save_input_master()

        viterbi_filenames = self.viterbi_filenames
        posterior_filenames = self.posterior_filenames

        # -: standard output, processed by segway-task
        viterbi_kwargs = dict(triFile=self.triangulation_filename,
                              pVitRegexFilter="^seg$",
                              pVitValsFile="-")

        posterior_kwargs = dict(triFile=self.posterior_triangulation_filename,
                                jtFile=self.posterior_jt_info_filename,
                                doDistributeEvidence=True,
                                **self.get_posterior_clique_print_ranges())

        # XXX: kill submitted jobs on exception
        with Session() as session:
            self.session = session

            restartable_jobs = RestartableJobDict(session, self.job_log_file)

            for window_index, window_len in self.window_lens_sorted():
                queue_identify_custom = partial(self._queue_identify,
                                                restartable_jobs, window_index)

                if (self.identify
                    and not self.recover_viterbi_window(window_index)):
                    queue_identify_custom(PREFIX_JOB_NAME_VITERBI,
                                          VITERBI_PROG, viterbi_kwargs,
                                          viterbi_filenames)

                if self.posterior:
                    queue_identify_custom(PREFIX_JOB_NAME_POSTERIOR,
                                          POSTERIOR_PROG, posterior_kwargs,
                                          posterior_filenames)

            # XXX: ask on DRMAA mailing list--how to allow
            # KeyboardInterrupt here?

            if self.dry_run:
                return

            restartable_jobs.wait()

        if self.identify:
            self.concatenate_bed()
            bed_filename = self.bed_filename
            layer(bed_filename, make_layer_filename(bed_filename),
                  bigbed_outfilename=self.bigbed_filename)

        if self.posterior:
            self.concatenate_bedgraph()

    def make_script_filename(self, prefix):
        return self.make_filename(prefix, EXT_SH, subdirname=SUBDIRNAME_LOG)

    def make_run_msg(self):
        now = datetime.now()
        pkg_desc = working_set.find(Requirement.parse(PKG))
        run_msg = "## %s run %s at %s" % (pkg_desc, UUID, now)

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

                        if self.posterior and self.recover_dirname:
                            raise NotImplementedError # XXX

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
        group.add_option("-D", "--distribution", choices=DISTRIBUTIONS,
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

        group.add_option("--prior-strength", type=float, metavar="RATIO",
                         help="use RATIO times the number of data counts as"
                         " the number of pseudocounts for the segment length"
                         " prior (default %f)" % PRIOR_STRENGTH)

        group.add_option("--segtransition-weight-scale", type=float,
                         metavar="SCALE",
                         help="exponent for segment transition probability "
                         " (default %f)" % SEGTRANSITION_WEIGHT_SCALE)

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
