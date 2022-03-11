#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        with_statement)

"""
run: main Segway implementation
"""

# Copyright 2008-2014 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from csv import writer
from datetime import datetime
from distutils.spawn import find_executable
from errno import EEXIST, ENOENT
from functools import partial
from itertools import count, product
from math import ceil, ldexp
from os import (environ, extsep, fdopen, open as os_open, O_WRONLY, O_CREAT,
                O_SYNC, O_TRUNC)
import re
from shutil import copy2
import stat
from string import ascii_letters
import sys
from threading import Lock, Thread
from time import sleep
from uuid import uuid1
from warnings import warn

from genomedata import Genome
from numpy import (append, arcsinh, array, empty, finfo, float32, int64, inf,
                   square, vstack, zeros)
from numpy.random import RandomState
from operator import attrgetter
from optplus import str2slice_or_int
from optbuild import AddableMixin
from path import Path
import pipes
from pkg_resources import parse_version, Requirement, working_set
from six import PY2, viewitems, viewvalues
from six.moves import map, range, zip
from six.moves.urllib.parse import quote
from tabdelim import DictReader, ListWriter

from .bed import parse_bed4, read_native
from .cluster import (is_running_locally, make_native_spec, JobTemplateFactory,
                      RestartableJob, RestartableJobDict, Session)
from .include import IncludeSaver
from .input_master import InputMasterSaver
from .observations import Observations
from .output import IdentifySaver, PosteriorSaver
from .structure import StructureSaver
from .task import MSG_SUCCESS
from ._util import (ceildiv, data_filename, DTYPE_OBS_INT, DISTRIBUTION_NORM,
                    DISTRIBUTION_GAMMA, DISTRIBUTION_ASINH_NORMAL,
                    DELIMITER_BED,
                    EXT_BED, EXT_FLOAT, EXT_GZ, EXT_INT, EXT_PARAMS,
                    EXT_TAB, extjoin, extjoin_not_none, GB,
                    ISLAND_BASE_NA, ISLAND_LST_NA, load_coords,
                    make_default_filename, make_filelistpath,
                    make_prefix_fmt, MB, memoized_property,
                    OFFSET_START, OFFSET_END, OFFSET_STEP,
                    OptionBuilder_GMTK, POSTERIOR_PROG,
                    PREFIX_LIKELIHOOD, PREFIX_PARAMS,
                    PREFIX_VALIDATION_OUTPUT,
                    PREFIX_VALIDATION_SUM,
                    PREFIX_VALIDATION_OUTPUT_WINNER,
                    PREFIX_VALIDATION_SUM_WINNER,
                    SEG_TABLE_WIDTH,
                    SUBDIRNAME_LOG, SUBDIRNAME_PARAMS, SUBDIRNAME_INTERMEDIATE,
                    SUPERVISION_LABEL_OFFSET,
                    SUPERVISION_UNSUPERVISED,
                    SUPERVISION_SEMISUPERVISED, USE_MFSDG,
                    VALIDATE_PROG, VITERBI_PROG,
                    VIRTUAL_EVIDENCE_LIST_FILENAME,
                    VIRTUAL_EVIDENCE_LIST_FILENAME_PLACEHOLDER)
from .version import __version__

# GMTK runtime requirements
MINIMUM_GMTK_VERSION = parse_version("1.4.2")
GMTK_VERSION_ERROR_MSG = """
GMTK version %s was detected.
Segway requires GMTK version %s or later to be installed.
Please update your GMTK version."""

# set once per file run
UUID = uuid1().hex

# XXX: I should really get some sort of Enum for this, I think Peter
# Norvig has one
DISTRIBUTIONS = [DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                 DISTRIBUTION_ASINH_NORMAL]
DISTRIBUTION_DEFAULT = DISTRIBUTION_ASINH_NORMAL

NUM_GAUSSIAN_MIX_COMPONENTS_DEFAULT = 1

MIN_NUM_SEGS = 2
NUM_SEGS = MIN_NUM_SEGS
NUM_SUBSEGS = 1
OUTPUT_LABEL = "seg"
RULER_SCALE = 10
MAX_EM_ITERS = 100
CARD_SUPERVISIONLABEL_NONE = -1
MINIBATCH_DEFAULT = -1
VAR_FLOOR_GMM_DEFAULT = 0.00001

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

JOIN_TIMEOUT = None
if PY2:
    JOIN_TIMEOUT = finfo(float).max

SWAP_ENDIAN = False

# option defaults
VERBOSITY = 0
PRIOR_STRENGTH = 0

FINFO_FLOAT32 = finfo(float32)
MACHEP_FLOAT32 = FINFO_FLOAT32.machep
TINY_FLOAT32 = FINFO_FLOAT32.tiny

SIZEOF_FLOAT32 = float32().nbytes
SIZEOF_DTYPE_OBS_INT = DTYPE_OBS_INT().nbytes

# sizeof tmp space used per frame
SIZEOF_FRAME_TMP = (SIZEOF_FLOAT32 + SIZEOF_DTYPE_OBS_INT)

FUDGE_EP = -17  # ldexp(1, -17) = ~1e-6
assert FUDGE_EP > MACHEP_FLOAT32

FUDGE_TINY = -ldexp(TINY_FLOAT32, 6)

LOG_LIKELIHOOD_DIFF_FRAC = 1e-5

NUM_SEQ_COLS = 2   # dinucleotide, presence_dinucleotide

NUM_SUPERVISION_COLS = 2  # supervision_data, presence_supervision_data

NUM_VIRTUAL_EVIDENCE_COLS = 1  # presence virtual evidence data

MAX_SPLIT_SEQUENCE_LENGTH = 2000000  # 2 million
MAX_FRAMES = MAX_SPLIT_SEQUENCE_LENGTH
MEM_USAGE_BUNDLE = 100 * MB  # XXX: should start using this again
MEM_USAGE_PROGRESSION = "2,3,4,6,8,10,12,14,15"

TMP_USAGE_BASE = 10 * MB  # just a guess

# train_window object has structure (window number, number of bases)
TRAIN_WINDOWS_WINDOW_NUM_INDEX = 0
TRAIN_WINDOWS_BASES_INDEX = 1
MAX_GMTK_WINDOW_COUNT = 10000

POSTERIOR_CLIQUE_INDICES = dict(p=1, c=1, e=1)

# defaults
NUM_INSTANCES = 1

CPP_DIRECTIVE_FMT = "-D%s=%s"

GMTK_INDEX_PLACEHOLDER = "@D"
NAME_BUNDLE_PLACEHOLDER = "bundle"

# programs
ENV_CMD = "/usr/bin/env"

# XXX: need to differentiate this (prog) from prog.prog == progname throughout
TRIANGULATE_PROG = OptionBuilder_GMTK("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_GMTK("gmtkEMtrain")

# Tasks
# 'verb' refers to the action being run, and 'kind' refers
# to a noun describing the task
ANNOTATE_TASK_KIND = "viterbi"
POSTERIOR_TASK_KIND = "posterior"
TRAIN_TASK_KIND = "train"
VALIDATE_TASK_VERB = "run"
VALIDATE_TASK_KIND = "validate"
SAVE_WINDOW_TASK_VERB = "save"
SAVE_WINDOW_TASK_KIND = "gmtk-observation-files"
BUNDLE_TRAIN_TASK_KIND = "bundle-train"

TMP_OBS_PROGS = frozenset([VITERBI_PROG, POSTERIOR_PROG])

# supported segway tasks
TASK_LIST = ["train", "identify", "posterior"]

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
PREFIX_TRAIN_RESULTS = "train_results"
PREFIX_POSTERIOR = "posterior%s"
PREFIX_IDENTIFY = "identify"

PREFIX_VITERBI = "viterbi"
PREFIX_WINDOW = "window"
PREFIX_JT_INFO = "jt_info"

# "job.{instance_name}.tab"
PREFIX_JOB_LOG = "jobs.{}"

PREFIX_JOB_NAME_TRAIN = "emt"
PREFIX_JOB_NAME_VITERBI = "vit"
PREFIX_JOB_NAME_POSTERIOR = "jt"
PREFIX_JOB_NAME_VALIDATE = "validate"
PREFIX_JOB_NAMES = dict(identify=PREFIX_JOB_NAME_VITERBI,
                        posterior=PREFIX_JOB_NAME_POSTERIOR)

SUFFIX_OUT = extsep + EXT_OUT
SUFFIX_TRIFILE = extsep + EXT_TRIFILE

# "segway.bed.gz"
BED_FILEBASENAME = extjoin(__package__, EXT_BED, EXT_GZ)

# "segway.%d.bed.gz"
BED_FILEBASEFMT = extjoin(__package__, "%d", EXT_BED, EXT_GZ)

# "posterior%s.bed.gz"
BEDGRAPH_FILEBASENAME = extjoin(PREFIX_POSTERIOR, EXT_BEDGRAPH, EXT_GZ)

# "posterior%s.%%d.bed.gz"
BEDGRAPH_FILEBASEFMT = extjoin(PREFIX_POSTERIOR, "%%d", EXT_BEDGRAPH, EXT_GZ)
FLOAT_TABFILEBASENAME = extjoin("observations", EXT_TAB)
TRAIN_FILEBASENAME = extjoin(PREFIX_TRAIN, EXT_TAB)
TRAIN_RESULTS_TMPL = extjoin(PREFIX_TRAIN_RESULTS, EXT_TAB, "tmpl")
IDENTIFY_FILEBASENAME = extjoin(PREFIX_IDENTIFY, EXT_TAB)

SUBDIRNAME_ACC = "accumulators"
SUBDIRNAME_AUX = "auxiliary"
SUBDIRNAME_JOB_SCRIPT = "cmdline"
SUBDIRNAME_LIKELIHOOD = "likelihood"
SUBDIRNAME_OBS = "observations"
SUBDIRNAME_VALIDATION = "validation"
SUBDIRNAME_POSTERIOR = "posterior"
SUBDIRNAME_VITERBI = "viterbi"

SUBDIRNAMES_EITHER = [SUBDIRNAME_AUX]
SUBDIRNAMES_TRAIN = [SUBDIRNAME_ACC, SUBDIRNAME_LIKELIHOOD,
                     SUBDIRNAME_PARAMS, SUBDIRNAME_INTERMEDIATE]

# job script file permissions: owner has read/write/execute
# permissions, group/others have read/execute permissions
JOB_SCRIPT_FILE_OPEN_FLAGS = O_WRONLY | O_CREAT | O_SYNC | O_TRUNC
JOB_SCRIPT_FILE_PERMISSIONS = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP \
                              | stat.S_IROTH | stat.S_IXOTH


JOB_LOG_FIELDNAMES = ["jobid", "jobname", "prog", "num_segs",
                      "num_frames", "maxvmem", "cpu", "exit_status"]
# XXX: should add num_subsegs as well, but it's complicated to pass
# that data into RestartableJobDict.wait()

TRAIN_FIELDNAMES = ["name", "value"]

TRAIN_OPTION_TYPES = \
    dict(input_master_filename=str, structure_filename=str,
         params_filename=str, dont_train_filename=str, seg_table_filename=str,
         include_coords_filename=str, distribution=str, len_seg_strength=float,
         num_instances=int, segtransition_weight_scale=float, ruler_scale=int,
         resolution=int, num_segs=int, num_subsegs=int, output_label=str,
         track_specs=[str], reverse_worlds=[int], num_mix_components=int,
         supervision_filename=str, virtual_evidence_filename=str, 
         minibatch_fraction=float, validation_fraction=float, max_frames=int,
         validation_coords_filename=str, var_floor=float)


TRAIN_RESULT_TYPES = dict(log_likelihood=float, num_segs=int,
                          validation_likelihood=float,
                          input_master_filename=str, params_filename=str,
                          log_likelihood_filename=str,
                          validation_output_filename=str,
                          validation_sum_filename=str)

IDENTIFY_OPTION_TYPES = \
    	dict(include_coords_filename=str, exclude_coords_filename=str,
             seg_table_filename=str, output_label=str, virtual_evidence_filename=str)

SUB_TASK_DELIMITER = "-"

# templates and formats
RES_OUTPUT_MASTER = "output.master"
RES_DONT_TRAIN = "dont_train.list"
RES_SEG_TABLE = "seg_table.tab"

COMMENT_POSTERIOR_TRIANGULATION = \
    "%% triangulation modified for posterior decoding by %s" % __package__

RESOLUTION = 1

SEGTRANSITION_WEIGHT_SCALE = 1.0
VIRTUAL_EVIDENCE_WEIGHT = 1.0

DIRPATH_WORK_DIR_HELP = Path("WORKDIR")
DIRPATH_AUX = DIRPATH_WORK_DIR_HELP / SUBDIRNAME_AUX
DIRPATH_PARAMS = DIRPATH_WORK_DIR_HELP / SUBDIRNAME_PARAMS

# 62 so that it's not in sync with the 10 second job wait sleep time
THREAD_START_SLEEP_TIME = 62  # XXX: this should become an option

# -gpr option for GMTK when reversing
REVERSE_GPR = "^0:-1:0"

OFFSET_FILENAMES = 3  # where the filenames begin in Results
OFFSET_VALIDATION_FILENAMES = 6  # where the validation filenames begin

GMTKJT_OUTPUT_PATTERN = \
    "Segment (\d+), after Prob E: log\(prob\(evidence\)\) = (\d+\.\d{1,9})?"
GMTKJT_PATTERN_WINDOW_MATCH_INDEX = 1
GMTKJT_PATTERN_LIKELIHOOD_MATCH_INDEX = 2

PROGS = dict(identify=VITERBI_PROG, posterior=POSTERIOR_PROG)


# functions
def quote_trackname(text):
    # legal characters are ident in GMTK_FileTokenizer.ll:
    # {alpha})({alpha}|{dig}|\_|\-)* (alpha is [A-za-z], dig is [0-9])
    res = text.replace("_", "_5F")
    res = res.replace(".", "_2E")

    # quote eliminates everything that doesn't match except for "_.-",
    # replaces with % escapes
    res = quote(res, safe="")  # safe="" => quote "/" too
    res = res.replace("%", "_")

    # add stub to deal with non-alphabetic first characters
    if res[0] not in ascii_letters:
        # __ should never appear in strings quoted as before
        res = "x__" + res

    return res


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
                print(inline, file=outfile)
            elif outline is None:
                outline = inline

        print(outline, file=outfile)

        while isinstance(outline, NoAdvance):
            outline = (yield)

            if outline is not None:
                print(outline, file=outfile)


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

    return range(start, stop, step)


def file_or_string_to_string_list(file_or_string):
    """Returns a list that will either contain each line from a file object or
    the single string given."""

    # Try to read all the lines from a file object
    try:
        return [line.rstrip() for line in file_or_string.readlines()]
    # If the object isn't a file, return its own type (presumed string)
    except AttributeError:
        return [file_or_string]


def is_training_progressing(last_ll, curr_ll,
                            min_ll_diff_frac=LOG_LIKELIHOOD_DIFF_FRAC):
    # using x !< y instead of x >= y to give the right default answer
    # in the case of NaNs
    return not abs((curr_ll - last_ll) / last_ll) < min_ll_diff_frac


def set_cwd_job_tmpl(job_tmpl):
    job_tmpl.workingDirectory = Path.getcwd()


def rewrite_cliques(rewriter, frame, output_label):
    """
    returns the index of the added clique
    """
    # method
    next(rewriter)

    # number of cliques
    orig_num_cliques = int(next(rewriter))
    rewriter.send(NoAdvance(orig_num_cliques + 1))

    # original cliques
    for clique_index in range(orig_num_cliques):
        next(rewriter)

    # new clique
    if output_label != "seg":
        rewriter.send(NewLine("%d 2 seg %d subseg %d" %
                              (orig_num_cliques, frame, frame)))
    else:
        rewriter.send(NewLine("%d 1 seg %d" % (orig_num_cliques, frame)))

    # XXX: add subseg as a clique to report it in posterior

    return orig_num_cliques


def make_mem_req(mem_usage):
    # double usage at this point
    mem_usage_gibibytes = ceil(mem_usage / GB)

    return "%dG" % mem_usage_gibibytes


class Mixin_Lockable(AddableMixin):  # noqa
    def __init__(self, *args, **kwargs):
        self.lock = Lock()
        return AddableMixin.__init__(self, *args, **kwargs)


LockableDefaultDict = Mixin_Lockable + defaultdict


class TrainInstanceResults():
    """
    Each TrainInstanceResults instance stores the filenames from the winning
    rounds of training for each segway train instance. Additionally contains
    their log likelihood and validation likelihoods.
    """
    def get_filenames(self, validation=False):
        """
        Returns a dictionary, where the keys represent runner attributes
        containing the dst filenames and the values are the source filenames
        stored in the class. Validation represents whether the validation
        files exist and need to be copied as well.
        """
        res = {"input_master_filename": self.input_master_filename,
               "params_filename": self.params_filename,
               "log_likelihood_filename": self.log_likelihood_filename}
        if validation:
            res.update({"validation_output_filenames":
                             self.validation_output_filename,
                        "validation_sum_filename":
                             self.validation_sum_filename})
        return res

    def load_results(self, filename):
        with open(filename) as resultfile:
            reader = DictReader(resultfile)
            for line in reader:
                name = line["name"]
                value = line["value"]

                item_type = TRAIN_RESULT_TYPES[name]
                setattr(self, name, item_type(value))

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


class TrainThread(Thread):
    def __init__(self, runner, session, instance_index, num_segs):
        # keeps it from rewriting variables that will be used
        # later or in a different thread
        self.runner = copy(runner)

        self.session = session
        self.num_segs = num_segs
        self.instance_index = instance_index
        self.runner.input_master_filename = make_default_filename(
            InputMasterSaver.resource_name, runner.params_dirpath,
            instance_index)

        Thread.__init__(self)

    def run(self):
        self.runner.session = self.session
        self.runner.num_segs = self.num_segs
        self.runner.instance_index = self.instance_index

        # If this thread is running on a cluster management system
        if not is_running_locally():
            # Sleep this thread to not overload cluster management system if
            # not the first training instance
            sleep(THREAD_START_SLEEP_TIME*self.instance_index)

        with self.runner.open_job_log_file() as self.runner.job_log_file:
            self.result = self.runner.run_train_instance()


def maybe_quote_arg(text):
    """return quoted argument, adding backslash quotes

    XXX: would be nice if this were smarter about what needs to be
    quoted, only doing this when special characters or whitespace are
    within the arg

    XXX: Enclosing characters in double quotes (`"') preserves the
    literal value of all characters within the quotes, with the
    exception of `$', ``', `\', and, when history expansion is
    enabled, `!'. The characters `$' and ``' retain their special
    meaning within double quotes (*note Shell Expansions::). The
    backslash retains its special meaning only when followed by one of
    the following characters: `$', ``', `"', `\', or `newline'. Within
    double quotes, backslashes that are followed by one of these
    characters are removed. Backslashes preceding characters without a
    special meaning are left unmodified. A double quote may be quoted
    within double quotes by preceding it with a backslash. If enabled,
    history expansion will be performed unless an `!' appearing in
    double quotes is escaped using a backslash. The backslash
    preceding the `!' is not removed.

    """
    return '"%s"' % text.replace('"', r'\"')


def cmdline2text(cmdline=sys.argv):
    return " ".join(maybe_quote_arg(arg) for arg in cmdline)


def _log_cmdline(logfile, cmdline):
    quoted_cmdline_list = []
    for cmdarg in cmdline:
        # pipes.quote() will only quote arguments that need
        # quotes to be run on shell commandline
        # NOTE: move to shlex.quote() when moving to python 3.3
        quoted_cmdline_list.append(pipes.quote(str(cmdarg)))
    quoted_cmdline = ' '.join(quoted_cmdline_list)
    print(quoted_cmdline, file=logfile)


def check_overlapping_supervision_labels(start, end, chrom, coords):
    for coord_start, coord_end in coords[chrom]:
        if not (coord_start >= end or coord_end <= start):
            raise ValueError("supervision label %s(%s, %s) overlaps"
                             "supervision label %s(%s, %s)" %
                             (chrom, coord_start, coord_end,
                              chrom, start, end))


def remove_bash_functions(environment):
    """Removes all bash functions (patched after 'shellshock') from an dictionary
    environment"""
    # Explicitly not using a dictionary comprehension to support Python
    # 2.6 (or earlier)
    # All bash functions in an exported environment after the shellshock
    # patch start with "BASH_FUNC"
    return dict(((key, value)
                for key, value in viewitems(environment)
                if not key.startswith("BASH_FUNC")))


class Track(object):
    def __init__(self, name_unquoted, is_data=True):
        self.name_unquoted = name_unquoted
        self.is_data = is_data
        self.group = None
        self.index = None
        self.genomedata_name = None

    @memoized_property
    def name(self):
        return quote_trackname(self.name_unquoted)


TRACK_DINUCLEOTIDE = Track("dinucleotide", is_data=False)
TRACK_SUPERVISIONLABEL = Track("supervisionLabel", is_data=False)
TRACK_VIRTUAL_EVIDENCE = Track("virtualEvidence", is_data=False)

EXCLUDE_TRACKNAME_LIST = [
    TRACK_DINUCLEOTIDE.name_unquoted,
    TRACK_SUPERVISIONLABEL.name_unquoted,
    TRACK_VIRTUAL_EVIDENCE.name_unquoted 
]


class TrackGroup(list):
    def _set_group(self, item):
        assert item.group is None
        item.group = self
        return item

    def __init__(self, items=[]):
        return list.__init__(self, [self._set_group(item) for item in items])

    def __setitem__(self, index, item):
        if isinstance(index, slice):
            raise NotImplementedError

        return list.__setitem__(self, index, self._set_group(item))

    def append(self, item):
        return list.append(self, self._set_group(item))

    def extend(self, items):
        return list.extend(self, [self._set_group(item) for item in items])

    def insert(self, index, item):
        return list.insert(self, index, self._set_group(item))


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
        self.virtual_evidence_filename = None

        self.params_filename = None  # actual params filename for this instance
        self.params_filenames = []  # list of possible params filenames
        self.recover_dirname = None
        self.work_dirname = None
        self.log_likelihood_filename = None
        self.log_likelihood_tab_filename = None

        self.validation_output_filename = None
        self.validation_output_winner_filename = None
        self.validation_output_tab_filename = None

        self.validation_sum_filename = None
        self.validation_sum_winner_filename = None
        self.validation_sum_tab_filename = None

        self.obs_dirname = None
        self.validation_obs_dirname = None

        self.include_coords_filename = None
        self.exclude_coords_filename = None
        self.validation_coords_filename = None

        self.minibatch_fraction = MINIBATCH_DEFAULT
        self.validation_fraction = None

        self.posterior_clique_indices = POSTERIOR_CLIQUE_INDICES.copy()

        self.triangulation_filename_is_new = None

        self.supervision_coords = None
        self.supervision_labels = None

        self.virtual_evidence_coords = None
        self.virtual_evidence_priors = None

        self.card_supervision_label = -1

        # tracks: list: all the tracks used
        self.tracks = []

        # track_groups: list of lists: each one is a grouping of
        # tracks that are similar to each other. Only one track from
        # each track group is used at a time during inference. A set
        # of tracks across groups being used at once is a "world."
        self.track_groups = []

        # default is 0
        self.global_mem_usage = LockableDefaultDict(int)

        # data
        # a "window" is what GMTK calls a segment
        self.windows = None
        self.validation_windows = None
        self.mins = None
        self.track_specs = []

        # variables
        self.num_segs = NUM_SEGS
        self.num_subsegs = NUM_SUBSEGS
        self.output_label = OUTPUT_LABEL
        self.num_instances = NUM_INSTANCES
        self.len_seg_strength = PRIOR_STRENGTH
        self.distribution = DISTRIBUTION_DEFAULT
        self.max_em_iters = MAX_EM_ITERS
        self.max_split_sequence_length = MAX_SPLIT_SEQUENCE_LENGTH
        self.max_frames = MAX_FRAMES
        self.segtransition_weight_scale = SEGTRANSITION_WEIGHT_SCALE
        self.track_weight = None
        self.virtual_evidence_weight = VIRTUAL_EVIDENCE_WEIGHT
        self.ruler_scale = RULER_SCALE
        self.resolution = RESOLUTION
        self.reverse_worlds = []  # XXXopt: this should be a set
        self.supervision_label_range_size = 0
        self.num_mix_components = NUM_GAUSSIAN_MIX_COMPONENTS_DEFAULT
        self.var_floor = None

        # flags
        self.clobber = False
        self.recover_round = False
        self.validate = False
        self.dry_run = False
        self.verbosity = VERBOSITY

        self.__dict__.update(kwargs)

    class SubTaskSpecification(object):
        """
        SubTaskSpecification instance stores the segway task/subtask
        specification for the current session.
        """
        def __init__(self, selected, subtask=None):
            # Subtask contains the name of the subtask chosen
            # this time (init, run or finish)
            if subtask:
                # set all subtasks to False initially
                self.init = False
                self.run = False
                self.finish = False
                # Overwrite the selected subtask back to true
                setattr(self, subtask[0], True)
            else:
                # If no subtask is specified, all subtasks are set to match
                # the selected boolean variable
                self.init = selected
                self.run = selected
                self.finish = selected

        def __bool__(self):
            return any([self.init, self.run, self.finish])

        # XXX: This line was added for python 2-3 compatibility
        # May be removed when python 2 support is dropped
        __nonzero__ = __bool__


    def set_tasks(self, text):
        """
        Sets the tasks for Segway to run, as well as determining
        if any specific steps were specified. Else runs all steps
        for task.
        """
        selected_tasks = text.split("+")

        for task in selected_tasks:
            task = task.split(SUB_TASK_DELIMITER)
            task_name = task[0]
            if task_name == "annotate":
                task_name = "identify"
            subtask = task[1:]
            # Check if multiple subtasks are present,
            # meaning run-round was chosen
            if (len(subtask) == 2 and
               (subtask[0] == "run" and subtask[1] == "round")):
                self.recover_dirname = self.work_dirname
                self.recover_round = True
            elif len(subtask) >= 2:
                # Otherwise error if more than one subtask was specified
                raise ValueError("More than one subtask was specified at once")
            setattr(self, task_name, self.SubTaskSpecification(True, subtask))

        for task in TASK_LIST:
            if not hasattr(self, task):
                setattr(self, task, self.SubTaskSpecification(False))

    def set_option(self, name, value):
        # want to actually set the Runner option when optparse option
        # is numeric zero (int or float), but not if it is an empty
        # list, empty str, None, False, etc.
        if value or value == 0:
            setattr(self, name, value)

    options_to_attrs = {"recover": "recover_dirname",
                        "observations": "obs_dirname",
                        "bed": "bed_filename",
                        "semisupervised": "supervision_filename",
                        "bigBed": "bigbed_filename",
                        "include_coords": "include_coords_filename",
                        "exclude_coords": "exclude_coords_filename",
                        "minibatch_fraction": "minibatch_fraction",
                        "validation_fraction": "validation_fraction",
                        "validation_coords": "validation_coords_filename",
                        "num_instances": "num_instances",
                        "verbosity": "verbosity",
                        "split_sequences": "max_split_sequence_length",
                        "clobber": "clobber",
                        "dry_run": "dry_run",
                        "input_master": "input_master_filename",
                        "structure": "structure_filename",
                        "dont_train": "dont_train_filename",
                        "seg_table": "seg_table_filename",
                        "distribution": "distribution",
                        "prior_strength": "len_seg_strength",
                        "segtransition_weight_scale":
                            "segtransition_weight_scale",
                        "ruler_scale": "ruler_scale",
                        "resolution": "resolution",
                        "var_floor": "var_floor",
                        "mixture_components": "num_mix_components",
                        "num_labels": "num_segs",
                        "num_sublabels": "num_subsegs",
                        "output_label": "output_label",
                        "max_train_rounds": "max_em_iters",
                        "reverse_world": "reverse_worlds",
                        "track": "track_specs",
                        "trainable_params": "params_filenames",
                        "virtual_evidence": "virtual_evidence_filename",
                        "virtual_evidence_weight": "virtual_evidence_weight",
                        "track_weight": "track_weight"}

    @classmethod
    def fromargs(cls, task_spec, archives, traindir, annotatedir):
        """Parses the arguments (not options) that were given to segway"""
        res = cls()

        if annotatedir:
            res.work_dirname = annotatedir
        else:
            res.work_dirname = traindir

        res.genomedata_names = archives
        res.set_tasks(task_spec)

        try:
            res.load_train_options(traindir)
        except IOError as err:
            # train.tab use is optional
            if err.errno != ENOENT:
                raise

        return res

    def from_environment(self):
        # If there is a seed from the environment
        try:
            # Get the seed
            self.random_seed = int(environ["SEGWAY_RAND_SEED"])
        # Otherwise set no seed
        except KeyError:
            self.random_seed = None

        # Create a random number generator for this runner
        self.random_state = RandomState(self.random_seed)

    def add_track_group(self, tracknames):
        tracks = self.tracks
        track_group = TrackGroup()
        tracknames_unquoted = set(track.name_unquoted for track in tracks)

        # non-allowed special trackname
        if "supervisionLabel" in tracknames:
            raise ValueError("'supervisionLabel' trackname is internally "
                             "reserved and not allowed in supplied "
                             "tracknames")

        for trackname in tracknames:
            if trackname in tracknames_unquoted:
                raise ValueError("can't tie one track in multiple groups")

            track = Track(trackname)
            tracks.append(track)
            track_group.append(track)
            tracknames_unquoted.add(trackname)

        self.track_groups.append(track_group)

    @classmethod
    def fromoptions(cls, task_spec, archives, traindir, annotatedir, options):
        """This is the usual way a Runner is created.

        Calls Runner.fromargs() first.
        """

        res = cls.fromargs(task_spec, archives, traindir, annotatedir)
        res.from_environment()

        # Convert the options namespace into a dictionary
        options_dict = vars(options)

        # Add all options from this task into the Runner object
        for option in options_dict.keys():
            # multiple lists to one
            if option == "cluster_opt":
                res.user_native_spec = sum([opt.split(" ")
                                            for opt in options.cluster_opt], [])
                # No further processing required on this option, continue

            elif option == "mem_usage":
                mem_usage_list = list(map(float, options.mem_usage.split(",")))
                # XXX: should do a ceil first?
                # use int64 in case run.py is run on a 32-bit machine to
                # control 64-bit compute nodes
                res.mem_usage_progression = \
                    (array(mem_usage_list) * GB).astype(int64)
                # No further processing required on this option, continue

            # If the observations directory has been specified
            elif (option == "observations"
                  and options_dict[option]):
                # Stop segway and show a "not implemented error"
                # with description
                raise NotImplementedError(
                    "'--observations' option not used: "
                    "Segway only creates observations in a temporary directory"
                )

            # All other options need to be processed and saved to Runner
            # The task_spec and args positional arguents handled in
            # fromargs above
            elif not (option == "archives" or option == "task_spec" or
                      option == "traindir" or option == "annotatedir"):
                # Preprocess options
                # Convert any track files into a list of tracks
                if (option == "track"
                    and options_dict[option]):
                    track_specs = []
                    tracks = options_dict[option]
                    for file_or_string in tracks:
                        track_specs.extend(
                            file_or_string_to_string_list(file_or_string))

                    options_dict[option] = track_specs

                # Convert labels string into potential slice or an int
                # If num labels was specified
                if (option == "num_labels"
                    and options_dict[option]):
                    options.num_labels = str2slice_or_int(options_dict[option])

                # bulk copy options that need no further processing
                dst = cls.options_to_attrs[option]

                res.set_option(dst, options_dict[option])

        # If the resolution is set to a non-default value
        # And the ruler has not been set from the options
        if (res.resolution != RESOLUTION and
           ("ruler_scale" in options_dict and
            not options_dict["ruler_scale"])):
            # Set the ruler scale to 10x the non-default value
            res.ruler_scale = res.resolution * 10
        # Else if the ruler is not divisible by the resolution
        elif res.ruler_scale % res.resolution != 0:
            # Report the value error
            raise ValueError("The ruler (%d) is not divisible by the"
                             " resolution (%d)" %
                             (res.ruler_scale, res.resolution))

        res.max_frames = ceildiv(res.max_split_sequence_length, res.resolution)

        # track option processing
        for track_spec in res.track_specs:
            # local to each value of track_spec
            res.add_track_group(track_spec.split(","))

        # Check if the dinucleotide option has been specified
        if any(track.name == "dinucleotide" for track in res.tracks):
            # Stop segway and show a "not implemented error" with description
            raise NotImplementedError(
                "'dinucleotide' track option not supported"
            )

        if res.num_worlds > 1:
            res.check_world_fmt("bed_filename")
            res.check_world_fmt("bedgraph_filename")
            res.check_world_fmt("bigbed_filename")

        if (res.identify and
            res.verbosity > 0):
            warn("Running Segway in identify mode with non-zero verbosity"
                 " is currently not supported and may result in errors.")

        if (res.var_floor and
            res.var_floor < 0):
            raise ValueError("The variance floor cannot be less than 0")

        if (res.validation_fraction and
            res.validation_coords_filename):
            raise ValueError("Cannot specify validation sets using both"
                " fraction and explicit coordinates")

        if (res.validation_fraction and
            res.validation_fraction < 0):
            raise ValueError("The validation fraction cannot be less than 0")

        if (res.validation_fraction or
            res.validation_coords_filename):
            res.validate = True

        if (res.validation_fraction and
            (res.validation_fraction +
            res.minibatch_fraction > 1.0)):
            raise ValueError("The sum of the validation and "
                "minibatch fractions cannot be greater than 1")

        return res

    @memoized_property
    def triangulation_dirpath(self):
        res = self.work_dirpath / "triangulation"
        # Check to make sure this wasn't created by an ealier subtask
        if not Path(res).exists():
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
        return Path(self.work_dirname)

    @memoized_property
    def recover_dirpath(self):
        recover_dirname = self.recover_dirname
        if recover_dirname:
            return Path(recover_dirname)

        # recover_dirname is None or ""
        return None

    @memoized_property
    def include_coords(self):
        return load_coords(self.include_coords_filename)

    @memoized_property
    def exclude_coords(self):
        return load_coords(self.exclude_coords_filename)

    @memoized_property
    def validation_coords(self):
        return load_coords(self.validation_coords_filename)

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

            # Ensure that all ruler lengths are equal
            first_ruler_length = None

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

                # When no step is specified, use the set ruler
                # NB: The default segment table does not have a step specified.
                segment_length_step = len_slice.step
                if segment_length_step is None:
                    segment_length_step = ruler_scale

                # When no minimum length is set, use the set ruler
                minimum_segment_length = len_slice.start
                if minimum_segment_length is None:
                    minimum_segment_length = ruler_scale

                # If this is the first row, get the first ruler length
                if first_ruler_length is None:
                    first_ruler_length = segment_length_step
                # Else if the first ruler length does not match this row's
                # length
                elif first_ruler_length is not segment_length_step:
                    # Raise a value error
                    raise ValueError("Segment table rulers are not equal."
                                     " Found ruler lengths %d and %d" %
                                     (first_ruler_length, segment_length_step))

                len_tuple = (minimum_segment_length, len_slice.stop,
                             segment_length_step)
                len_row = zeros((SEG_TABLE_WIDTH))

                for item_index, item in enumerate(len_tuple):
                    if item is not None:
                        len_row[item_index] = item

                res[row_indexes] = len_row

        return res

    @memoized_property
    def validation_obs_dirpath(self):
        res = self.work_dirpath / SUBDIRNAME_OBS / SUBDIRNAME_VALIDATION
        self.validation_obs_dirname = res
        try:
            self.make_dir(res)
        except OSError as err:
            if not (err.errno == EEXIST and res.isdir()):
                raise

        return res

    @memoized_property
    def obs_dirpath(self):
        obs_dirname = self.obs_dirname

        if obs_dirname:
            res = Path(obs_dirname)
        else:
            res = self.work_dirpath / SUBDIRNAME_OBS
            self.obs_dirname = res

        # XXX: Disable creating the observation directory until the observation
        # option is re-enabled.
        # XXX: The following 3 properties are technically unused as a result
        # try:
        #     self.make_dir(res)
        # except OSError, err:
        #     if not (err.errno == EEXIST and res.isdir()):
        #         raise

        return res

    @memoized_property
    def float_filelistpath(self):
        return self.make_obs_filelistpath(EXT_FLOAT)

    @memoized_property
    def int_filelistpath(self):
        return self.make_obs_filelistpath(EXT_INT)

    @memoized_property
    def validation_float_filelistpath(self):
        return self.make_validation_obs_filelistpath(EXT_FLOAT)

    @memoized_property
    def validation_int_filelistpath(self):
        return self.make_validation_obs_filelistpath(EXT_INT)

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

        # self.gmtk_include_filename_relative = include_filename_relative

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
        return self.transform(sums_squares_normalized -
                              square(self._means_untransformed))

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

        viterbi_filename_fmt = (PREFIX_VITERBI + make_prefix_fmt(num_windows) +
                                EXT_BED)
        return [viterbi_dirpath / viterbi_filename_fmt % index
                for index in range(num_windows)]

    @memoized_property
    def viterbi_filenames(self):
        return self.make_viterbi_filenames(self.work_dirpath)

    @memoized_property
    def recover_viterbi_filenames(self):
        recover_dirpath = self.recover_dirpath
        if recover_dirpath:
            return self.make_viterbi_filenames(recover_dirpath)
        else:
            return None

    @memoized_property
    def posterior_filenames(self):
        return list(map(self.make_posterior_filename, range(self.num_windows)))

    @memoized_property
    def recover_posterior_filenames(self):
        raise NotImplementedError  # XXX

    @memoized_property
    def params_dirpath(self):
        return self.work_dirpath / SUBDIRNAME_PARAMS

    @memoized_property
    def intermediate_dirpath(self):
        return self.work_dirpath / SUBDIRNAME_INTERMEDIATE

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
        infilename_stem = (infilename.rpartition(SUFFIX_TRIFILE)[0] or
                           infilename)

        res = extjoin(infilename_stem, EXT_POSTERIOR, EXT_TRIFILE)

        clique_indices = self.posterior_clique_indices

        # XXX: this is a fairly hacky way of doing it and will not
        # work if the triangulation file changes from what GMTK
        # generates. probably need to key on tokens rather than lines
        with open(infilename) as infile:
            with open(res, "w") as outfile:
                print(COMMENT_POSTERIOR_TRIANGULATION, file=outfile)
                rewriter = rewrite_strip_comments(infile, outfile)

                consume_until(rewriter, "@@@!!!TRIFILE_END_OF_ID_STRING!!!@@@")
                consume_until(rewriter, "CE_PARTITION")

                components_indexed = enumerate(POSTERIOR_CLIQUE_INDICES)
                for component_index, component in components_indexed:
                    clique_index = rewrite_cliques(rewriter, component_index,
                                                   self.output_label)
                    clique_indices[component] = clique_index

                for line in rewriter:
                    pass

        return res

    def output_dirpath(self):
        return self.make_output_dirpath("o", self.instance_index)

    def error_dirpath(self):
        return self.make_output_dirpath("e", self.instance_index)

    def job_script_dirpath(self):
        return self.make_job_script_dirpath(self.instance_index)

    @contextmanager
    def open_job_log_file(self):
        """Create a job log file for a given instance."""
        # NB: Let exceptions raise on their own
        # Attempt to open the job log file
        job_log_filename = self.make_filename(PREFIX_JOB_LOG.format(
                                              self.instance_index),
                                              EXT_TAB,
                                              subdirname=SUBDIRNAME_LOG)

        with open(job_log_filename, "a") as job_log_file:
            # Print job log header if the file is new
            if job_log_file.tell() == 0:
                print(*JOB_LOG_FIELDNAMES, sep="\t", file=job_log_file)

            yield job_log_file

    @memoized_property
    def use_dinucleotide(self):
        return TRACK_DINUCLEOTIDE in self.tracks

    @memoized_property
    def num_int_cols(self):
        if not USE_MFSDG or self.resolution > 1:
            res = self.num_track_groups
        else:
            res = 0

        if self.use_dinucleotide:
            res += NUM_SEQ_COLS
        if self.supervision_type != SUPERVISION_UNSUPERVISED:
            res += NUM_SUPERVISION_COLS
        if self.virtual_evidence:
            res += NUM_VIRTUAL_EVIDENCE_COLS

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
    def validate_prog(self):
        return self.prog_factory(VALIDATE_PROG)

    @memoized_property
    def seg_countdowns_initial(self):
        table = self.seg_table

        starts = table[:, OFFSET_START]
        ends = table[:, OFFSET_END]
        steps = table[:, OFFSET_STEP]

        # XXX: need to assert that ends are either 0 or are always
        # greater than starts

        # starts and ends must all be divisible by steps
        if ((starts % steps).any() or
           (ends % steps).any()):
            raise ValueError("A segment table start or end boundary is not"
                             " divisible by the ruler (%d)" % steps[0])

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
    def num_track_groups(self):
        return len(self.track_groups)

    @memoized_property
    def num_windows(self):
        return len(self.windows)

    @memoized_property
    def num_bases(self):
        return sum(self.window_lens)

    @memoized_property
    def supervision_type(self):
        # Supervision is only used in training
        # If another task is selected, set to unsupervised
        if self.supervision_filename and self.train:
            return SUPERVISION_SEMISUPERVISED
        else:
            return SUPERVISION_UNSUPERVISED

    @memoized_property
    def virtual_evidence(self):
        if self.virtual_evidence_filename is not None:
            return True
        else:
            return False

    @memoized_property
    def world_track_indexes(self):
        """
        Track indexes for a particular world.
        """

        return [[track.index for track in world]
                for world in zip(*self.track_groups)]

    @memoized_property
    def world_genomedata_names(self):
        """
        Genomedata archive list for a world.
        Ordered based on track groups
        """

        return [[track.genomedata_name for track in world]
                for world in zip(*self.track_groups)]

    @memoized_property
    def num_worlds(self):
        # XXX: add support for some heads having only one trackname
        # that is repeated

        track_groups = self.track_groups
        if not track_groups:  # use all the data in the archive
            return 1

        res = len(track_groups[0])

        assert all(len(track_group) == res for track_group in track_groups)

        return res

    @memoized_property
    def instance_make_new_params(self):
        """
        should I make new parameters in each instance?
        """
        return self.num_instances > 1 or isinstance(self.num_segs, slice)

    @memoized_property
    def num_segs_range(self):
        return slice2range(self.num_segs)

    def check_world_fmt(self, attr):
        """ensure that all options that need a template have it
        """
        value = getattr(self, attr)
        if value is None:
            return

        try:
            value % 0
        except TypeError:
            raise TypeError("%s for use with multiple worlds must contain "
                            "one format string character such as %%s" % attr)

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

        # prevent supervised variable from being inherited from train task
        if self.identify or self.posterior:
            directives["CARD_SUPERVISIONLABEL"] = CARD_SUPERVISIONLABEL_NONE

        if self.virtual_evidence:
            directives[VIRTUAL_EVIDENCE_LIST_FILENAME] = \
                VIRTUAL_EVIDENCE_LIST_FILENAME_PLACEHOLDER
            directives["VIRTUAL_EVIDENCE"] = 1

        directives["CARD_SEG"] = self.num_segs
        directives["CARD_SUBSEG"] = self.num_subsegs
        directives["CARD_FRAMEINDEX"] = self.max_frames
        directives["SEGTRANSITION_WEIGHT_SCALE"] = \
            self.segtransition_weight_scale

        res = " ".join(CPP_DIRECTIVE_FMT % item
                       for item in viewitems(directives))

        if res:
            return res

        # default: return None

    def load_log_likelihood(self):
        with open(self.log_likelihood_filename) as infile:
            log_likelihood = infile.read().strip()

        with open(self.log_likelihood_tab_filename, "a") as logfile:
            print(log_likelihood, file=logfile)

        return float(log_likelihood)

    def load_validation_log_likelihood(self):
        """
        Parses gmtkJT output to obtain each window's log probability of
        evidence log(prob(E)).

        Raises error if it is found that the validation task was unsucccessful.

        Returns the validation window indices and corresponding likelihoods
        """
        with open(self.validation_output_filename) as infile:
            validation_output = infile.readlines()

        validation_window_indices = []
        validation_window_likelihoods = []

        for line in validation_output:
            # returns a regex group where first match is validation
            # window index, second match is validation window likelihood
            validation_match = re.search(GMTKJT_OUTPUT_PATTERN, line)
            if validation_match:
                validation_window_indices.append(int(
                    validation_match.group(GMTKJT_PATTERN_WINDOW_MATCH_INDEX)
                    ))
                validation_window_likelihoods.append(float(
                    validation_match.group(
                        GMTKJT_PATTERN_LIKELIHOOD_MATCH_INDEX)))
            else:
                if MSG_SUCCESS not in line:
                    raise RuntimeError("Validation not successful: " + line)

        return validation_window_indices, validation_window_likelihoods

    def get_full_validation_output(self, validation_window_indices,
                                   validation_window_likelihoods):
        """
        Takes as input the validation window indices and corresponding
        likelihoods and returns an array of tuples where each entry is the
        validation window index, size, and log likelihood.
        """
        full_validation_output = []
        for validation_window_index, validation_window_likelihood \
                in zip(validation_window_indices,
                       validation_window_likelihoods):
            validation_window = \
                self.validation_windows[validation_window_index]
            validation_window_size = len(validation_window)

            full_validation_output.append((
                validation_window_index,
                validation_window_size,
                validation_window_likelihood
                ))

        return full_validation_output

    def calculate_validation_log_likelihood(self, validation_window_likelihoods):
        """
        Takes as input an array of validation window likelihoods
        and sums to obtain the log of the intersection of the partial
        prob(E)'s (ie prob(E_1 and E_2 and ...))

        Returns the full validation set log(prob(E)).
        """
        return sum(validation_window_likelihoods)

    def make_filename(self, *exts, **kwargs):
        """
        makes a filename by joining together exts

        kwargs:
        dirname: top level directory (default self.work_dirname)
        subdirname: next level directory
        """
        filebasename = extjoin_not_none(*exts)

        # add subdirname if it exists
        return Path(kwargs.get("dirname", self.work_dirname)) \
            / kwargs.get("subdirname", "") \
            / filebasename

    def set_tracknames(self):
        """Set up track groups if not done already.

        Add index in Genomedata file for each data track.
        """

        tracks = self.tracks
        is_tracks_from_archive = False

        # Create a list of all tracks from all archives specified
        all_genomedata_tracks = []
        for genomedata_name in self.genomedata_names:
            with Genome(genomedata_name) as genome:
                all_genomedata_tracks.extend(genome.tracknames_continuous)

        # If tracks were specified on the command line or from file
        if tracks:
            # Check that all tracks specified appear only once across all
            # archives
            for trackname in (track.name_unquoted for track in tracks
                              if track.name_unquoted not in
                              EXCLUDE_TRACKNAME_LIST):
                track_count = all_genomedata_tracks.count(trackname)
                if track_count != 1:
                    raise ValueError(
                        "Track: {0} was found {1} times across all"
                        " archives".format(trackname, track_count)
                    )
        # Otherwise by default add all tracks from all archives
        else:
            # Add all tracks into individual track groups
            is_tracks_from_archive = True
            for trackname in all_genomedata_tracks:
                self.add_track_group([trackname])  # Adds to self.tracks

        # Raise an error if there are overlapping track names
        if self.is_tracknames_unique():
            error_msg = ""
            if is_tracks_from_archive:
                error_msg = "Duplicate tracknames found across genomedata"
                " archives"
            else:
                error_msg = "Arguments have duplicate tracknames"

            raise ValueError(error_msg)

        # For every data track in the track list
        for data_track in (track for track in tracks
                           if track.is_data):
            # For every genomedata archive
            for genomedata_name in self.genomedata_names:
                with Genome(genomedata_name) as genome:
                    # If the data track exists in this genome
                    if (data_track.name_unquoted in
                       genome.tracknames_continuous):
                        # Record the index of each data track specified found
                        # in this genome
                        data_track.index = genome.index_continuous(
                            data_track.name_unquoted
                        )
                        data_track.genomedata_name = genomedata_name
                        # Stop looking for this data track
                        break

        # If all tracks are not data
        if not any(track.is_data for track in tracks):
            # Set the float file list path to None
            self.float_filelistpath = None

    def get_last_params_filename(self, params_filename):
        if params_filename is not None and Path(params_filename).exists():
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
        elif (instance_index is not None and
              num_params_filenames > instance_index):
            params_filename = params_filenames[instance_index]
        else:
            params_filename = None

        last_params_filename = self.get_last_params_filename(params_filename)

        # make new filenames when new is set, or params_filename is
        # not set, or the file already exists and we are training
        if (new or not params_filename or
           (self.train and Path(params_filename).exists())):
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

    def make_validation_sum_tab_filename(self, instance_index, dirname):
        return self.make_filename(PREFIX_VALIDATION_SUM,
                                  instance_index, EXT_TAB,
                                  dirname=dirname,
                                  subdirname=SUBDIRNAME_LOG)

    def make_validation_output_tab_filename(self, instance_index,
                                                 dirname):
        return self.make_filename(PREFIX_VALIDATION_OUTPUT, instance_index,
                                  EXT_TAB, dirname=dirname,
                                  subdirname=SUBDIRNAME_LOG)

    def set_log_likelihood_filenames(self, instance_index=None, new=False):
        """
        None means the final params file, not for any particular thread
        """
        if new or not self.log_likelihood_filename:
            log_likelihood_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, instance_index,
                                   EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.log_likelihood_filename = log_likelihood_filename

            self.log_likelihood_tab_filename = \
                self.make_log_likelihood_tab_filename(instance_index,
                                                      self.work_dirname)

    def set_validation_likelihood_filenames(self, instance_index=None,
                                            new=False):
        """
        If no instance_index specified and the number of instances is
        greater than 1, then this is for the final validation
        likelihood files, not for any particular thread
        """
        if new or not self.validation_output_filename:
            # create validation output files
            validation_output_filename = \
                self.make_filename(PREFIX_VALIDATION_OUTPUT,
                                   instance_index, EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.validation_output_filename = \
                validation_output_filename

            validation_output_winner_filename = \
                self.make_filename(PREFIX_VALIDATION_OUTPUT_WINNER,
                                   instance_index, EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.validation_output_winner_filename = \
                validation_output_winner_filename

            self.validation_output_tab_filename = \
                self.make_validation_output_tab_filename(
                    instance_index, self.work_dirname)

            # create validation sum files
            validation_sum_filename = \
                self.make_filename(PREFIX_VALIDATION_SUM,
                                   instance_index, EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.validation_sum_filename = \
                validation_sum_filename

            validation_sum_winner_filename = \
                self.make_filename(PREFIX_VALIDATION_SUM_WINNER,
                                   instance_index, EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.validation_sum_winner_filename = \
                validation_sum_winner_filename

            self.validation_sum_tab_filename = \
                self.make_validation_sum_tab_filename(instance_index,
                                                      self.work_dirname)

    def make_output_dirpath(self, dirname, instance_index):
        res = self.work_dirpath / "output" / dirname / str(instance_index)
        # This is called every time Segway requires an output_dirpath
        # so no need to create the directory if it already exists
        if not Path(res).exists():
            self.make_dir(res)
        return res

    def make_job_script_dirpath(self, instance_index):
        res = self.work_dirpath / SUBDIRNAME_JOB_SCRIPT / str(instance_index)
        # This is called every time Segway requires a job script dirpath
        # so no need to create the directory if it already exists
        if not Path(res).exists():
            self.make_dir(res)
        return res

    def make_dir(self, dirname, clobber=None):
        if clobber is None:
            clobber = self.clobber

        dirpath = Path(dirname)

        if clobber:
            # just always try to delete it
            try:
                dirpath.rmtree()
            except OSError as err:
                if err.errno != ENOENT:
                    raise

        try:
            dirpath.makedirs()
        except OSError as err:
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

    def make_validation_obs_filelistpath(self, ext):
        return make_filelistpath(self.validation_obs_dirpath, ext)

    def save_resource(self, resname, subdirname=""):
        orig_filename = data_filename(resname)

        orig_filepath = Path(orig_filename)
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

    def subset_metadata_attr(self, name, reducer=sum):
        """subset a single metadata attribute to only the used tracks,
        grouping things in the same track groups together
        """

        # Get the data type for this genomedata attribute
        with Genome(self.genomedata_names[0]) as genome:
            attr = getattr(genome, name)

        track_groups = self.track_groups
        # Create an empty list of the same type as the genomedata attribute and
        # of the the same length of the track groups
        shape = len(track_groups)
        subset_array = empty(shape, attr.dtype)

        # For every track group
        for track_group_index, track_group in enumerate(track_groups):
            # For every genomedata archive
            track_group_attributes = empty(0, attr.dtype)
            for genomedata_name in self.genomedata_names:
                with Genome(genomedata_name) as genome:
                    # For each track in the track group that is in this archive
                    # Add the attribute from the track into a list
                    track_indexes = [track.index for track in track_group
                                     if track.genomedata_name ==
                                     genomedata_name]
                    genome_attr = getattr(genome, name)
                    track_group_attributes = append(track_group_attributes,
                                                    genome_attr[track_indexes])

            # Apply the "reducer" to the attributes list for this track group
            subset_array[track_group_index] = reducer(track_group_attributes)

        setattr(self, name, subset_array)

    def subset_metadata(self):
        """limits all the metadata attributes to only tracks that are used
        """
        subset_metadata_attr = self.subset_metadata_attr
        subset_metadata_attr("mins", min)
        subset_metadata_attr("sums")
        subset_metadata_attr("sums_squares")
        subset_metadata_attr("num_datapoints")

    def save_input_master(self, instance_index=None, new=False):
        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        _, input_master_filename_is_new = \
            InputMasterSaver(self)(input_master_filename, self.params_dirpath,
                                   self.clobber, instance_index)

    def load_supervision(self):
        # The semi-supervised mode changes the DBN structure so there is an
        # additional "supervisionLabel" observed variable at every frame. This
        # can then be used to deterministically force the hidden state to have
        # a certain value at certain positions. From the Segway application's
        # point of view, absolutely everything else is the same--it works just
        # like unsupervised segmentation with an extra observed variable and
        # slightly different. GMTK does all the magic here (and from GMTK's
        # point of view, it's just a slightly different structure with a
        # different observed variable).
        #
        # The *semi* part of semi-supervised is that you can do this only at
        # certain positions and leave the rest of the genome unsupervised.

        # defaultdict of list of ints
        # key: chrom
        # value: list of ints (label as number)
        supervision_labels = defaultdict(list)

        # defaultdict of lists of tuples
        # key: chrom
        # value: list of tuples (start, end)
        supervision_coords = defaultdict(list)

        with open(self.supervision_filename) as supervision_file:
            for datum in read_native(supervision_file):
                chrom = datum.chrom
                start = datum.chromStart
                end = datum.chromEnd

                check_overlapping_supervision_labels(start, end, chrom,
                                                     supervision_coords)

                supervision_coords[chrom].append((start, end))

                name = datum.name
                start_label, breaks, end_label = name.partition(":")
                if end_label == "":
                    # Supervision label specified without the colon
                    # This means it's not a soft assignment e.g "0" which
                    # means 0 is supervision label and extension is 1
                    # since we only need to keep 1 label.
                    supervision_labels[chrom].append(int(start_label))
                    self.set_supervision_label_range_size(1)
                else:
                    # Supervision label specified with a colon
                    # This means it's a soft assignment e.g. "0:5" which
                    # allow supervision label to be in [0,5). Here we should
                    # use 0 as supervison label and extension=5-0=5 meaning
                    # we keep 5 kind of labels (0,1,2,3,4).
                    start_label = int(start_label)
                    end_label = int(end_label)
                    supervision_labels[chrom].append(start_label)
                    if end_label <= start_label:
                        raise ValueError(
                            "Supervision label end label must be greater "
                            "than start label."
                        )
                    self.set_supervision_label_range_size(end_label -
                                                          start_label)

        max_supervision_label = max(max(labels)
                                    for labels
                                    in viewvalues(supervision_labels))

        self.supervision_coords = supervision_coords
        self.supervision_labels = supervision_labels

        self.tracks.append(TRACK_SUPERVISIONLABEL)
        self.card_supervision_label = (max_supervision_label + 1 +
                                       SUPERVISION_LABEL_OFFSET)

    def load_virtual_evidence(self):
        # Virtual evidence adds one additional binary node C to the DBN structure
        # at every frame. A CPT is defined for C such that P(C|A=a) is the probability
        # that the parent A (segment label) is a particular value 'a' at a given frame.
        # Specifically, C exists only to constrain the parent (segment label) to be a 
        # particular value (supervised label) with the specified prior probability.
        # From the Segway application's point of view, nothing more changes
        # other than the virtual evidence files need to be passed forwards to GMTK,
        # which does all the magic here.
        #
        # It is possible to specify virtual evidence for some positions and not others.
        # This is handled using a presence variable for virtual evidence, which behaves
        # identically to other presence variables in Segway.
        if not self.virtual_evidence:
            return

        # Check if file supplied during init exists, warn if not saying it will
        # have to be replaced before running.
        if not Path(self.virtual_evidence_filename).exists():
           if self.train.run:
               raise IOError("Could not locate virtual evidence file: %s"
                                       % (self.virtual_evidence_filename))
           elif self.train.init or self.identify.init or self.posterior.init:
                warn("Virtual evidence file provided does not exist."
                     " A new one will need to be supplied by --virtual-evidence"
                     " during train-run for Segway to be able to apply it to the model")
           return
        # defaultdict of list of dictionaries of the format {label: prior} 
        # key: chrom
        # value: list of dictionaries of the format {label: prior}
        virtual_evidence_priors = defaultdict(list)

        # defaultdict of lists of tuples
        # key: chrom
        # value: list of tuples (start, end)
        virtual_evidence_coords = defaultdict(list)

        with open(self.virtual_evidence_filename) as ve_file:
            for datum in read_native(ve_file):
                chrom = datum.chrom
                start = datum.chromStart
                end = datum.chromEnd
                label = datum.name
                prior = datum.score

                # Check if a world was specified for this prior
                if hasattr(datum, "strand"):
                    worlds = [int(datum.strand)]
                # Otherwise apply it to all worlds be trained on
                else:
                    worlds = range(self.num_worlds)

                for world in worlds:
                    virtual_evidence_coords[(world, chrom)].append(
                        (start, end))
                    virtual_evidence_priors[(world, chrom)].append(
                        {int(label): float(prior)})

        self.virtual_evidence_coords = virtual_evidence_coords
        self.virtual_evidence_priors = virtual_evidence_priors
    
    def get_virtual_evidence_in_window(self, window):
        """Returns a tuple of a list of coords and their cooresponding
        dictionary of priors based on the region described the window"""

        # Return no virtual evidence if it is not enabled
        virtual_evidence_coords = None
        virtual_evidence_priors = None
        # Otherwise
        if self.virtual_evidence:
            virtual_evidence_coords = []
            virtual_evidence_priors = []
            # Iterate over all VE coordinates and corresponding priors
            zipper = zip(
                self.virtual_evidence_coords[(window.world, window.chrom)],
                self.virtual_evidence_priors[(window.world, window.chrom)]
            )
            for ve_coord, ve_prior in zipper:
                # A VE coord does not overlap with the window if its interval
                # ends entirely before the start of the window or starts
                # entirely after the end of the window
                # e.g. ve_coord[1] <= window.start or ve_coord[0] > window.end
                # Negate that condition (so a VE coord does overlap) for the
                # following:
                if ve_coord[1] >= window.start and ve_coord[0] < window.end:
                    # On VE coordinate overlap with window
                    # Append to our resulting list
                    virtual_evidence_coords.append(ve_coord)
                    virtual_evidence_priors.append(ve_prior)
        
        # Return list of VE coords and corresponding priors
        return (virtual_evidence_coords, virtual_evidence_priors)
                
    def set_supervision_label_range_size(self, new_extension):
        """This function takes a new label extension new_extension,
        and add it into the supervision_label_range_size set.

        Currently supervision_label_range_size only support one value,
        thus assignment is used to represent add functionality.
        If a new value tries to override an existing value (that is being
        set and doesn't match), an error will be thrown.
        """
        current_extension = self.supervision_label_range_size
        if current_extension != 0 and current_extension != new_extension:
            raise NotImplementedError(
                "Semisupervised soft label assignment currently does "
                "not support different range sizes. "
            )
        else:
            self.supervision_label_range_size = new_extension

    def save_structure(self):
        self.structure_filename, _ = \
            StructureSaver(self)(self.structure_filename, self.work_dirname,
                                 self.clobber)

    def check_genomedata_archives(self):
        """ Checks that all genomedata archives have matching chromosomes and
        that each chromosome has matching start and end coordinates

        Returns True if archives are valid, False otherwise"""

        # TODO: Ideally it's only necessary that for chromosomes that do match,
        # that their start and end coords match. As in, they do not necessarily
        # have to match on a per-chromosome basis. However when we assume that
        # they do it makes locating windows trivial and only has to be done on
        # a single genome

        # If there's more than one genomedata archive
        if len(self.genomedata_names) > 1:

            # Open the first genomedata archive as a reference
            with Genome(self.genomedata_names[0]) as sbjct:
                sbjct_coords = {}
                for chromosome in sbjct:
                    coord = (chromosome.start, chromosome.end)
                    sbjct_coords[chromosome.name] = coord

                # For every other genomedata archive
                for genomedata_name in self.genomedata_names[1:]:
                    with Genome(genomedata_name) as genome:
                        for chromosome in genome:
                            # If this chromosome doesn't exist in the reference
                            # genomedata archive
                            if chromosome.name not in \
                               sbjct_coords.keys():
                                # Return false
                                return False
                            # If the start and end coords don't match for this
                            # chromosome
                            if sbjct_coords[chromosome.name] != (
                                chromosome.start, chromosome.end
                            ):
                                # Return false
                                return False

                # Otherwise we've completed all checks successfully
                # Return true
                return True

        # Otherwise return true (for a single archive)
        else:
            return True

    def is_tracknames_unique(self):
        """ Checks if there exists more than one track with the same name.

        Returns True if duplicates found. False otherwise. """
        tracks = self.tracks
        tracknames_quoted = [track.name for track in tracks]
        return len(tracknames_quoted) != len(frozenset(tracknames_quoted))

    def save_gmtk_input(self):
        if self.supervision_type == SUPERVISION_SEMISUPERVISED:
            self.load_supervision()
        self.load_virtual_evidence()
        self.set_tracknames()

        observations = Observations(self)

        # Use all genomedata archives for locating windows
        try:
            genomes = [Genome(genomedata_name) for genomedata_name in
                       self.genomedata_names]

            observations.locate_windows(genomes)
        finally:
            for genome in genomes:
                genome.close()

        self.windows = observations.windows
        self.validation_windows = observations.validation_windows

        # XXX: does this need to be done before save()?
        self.subset_metadata()

        observations.create_validation_filepaths()
        self.validation_float_filepaths = \
            observations.validation_float_filepaths
        self.validation_int_filepaths = \
            observations.validation_int_filepaths

        if self.train:
            self.set_log_likelihood_filenames()

            if self.validate:
                self.set_validation_likelihood_filenames()

        self.save_include()
        self.set_params_filename()
        self.save_structure()

    def save_window_list(self):
        """Saves the current list of windows to a BED file where the name field
        is the window index that Segway has assigned"""

        window_bed_filename = self.make_filename(PREFIX_WINDOW, EXT_BED)

        with open(window_bed_filename, "w") as window_bed_file:
            bed_writer = writer(window_bed_file, delimiter=DELIMITER_BED,
                    lineterminator="\n")  #NB: csv defaults to dos newlines
            for index, window in enumerate(self.windows):
                bed_writer.writerow((window.chrom, window.start, window.end,
                                     index))

    def copy_results(self, name, src_filename):
        dst_filename = getattr(self, name)
        if dst_filename == src_filename:
            return
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
        return self.make_filename(PREFIX_POSTERIOR, window_index, EXT_BED,
                                  subdirname=SUBDIRNAME_POSTERIOR)

    def make_job_name_train(self, instance_index, round_index, window_index):
        return "%s%d.%d.%s.%s.%s" % (PREFIX_JOB_NAME_TRAIN, instance_index,
                                     round_index, window_index,
                                     self.work_dirpath.name, self.uuid)

    def make_job_name_validate(self, instance_index, round_index):
        return "%s%d.%d.%s.%s" % (PREFIX_JOB_NAME_VALIDATE, instance_index,
                                  round_index, self.work_dirpath.name,
                                  self.uuid)

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
                   obsNAN=True)

        if ISLAND:
            res["base"] = ISLAND_BASE
            res["lst"] = ISLAND_LST

        if HASH_LOAD_FACTOR is not None:
            res["hashLoadFactor"] = HASH_LOAD_FACTOR

        # XXX: dinucleotide-only won't work, because it has no float data
        assert (self.float_filelistpath and
                any(track.is_data for track in self.tracks))

        if self.float_filelistpath:
            res.update(of1=self.float_filelistpath,
                       fmt1="binary",
                       nf1=self.num_track_groups,
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
        zipper = sorted(zip(window_lens, count()), reverse=reverse)

        # XXX: use itertools instead of a generator
        for window_len, window_index in zipper:
            yield window_index, window_len

    def log_cmdline(self, cmdline, args=None, script_file=None):
        if args is None:
            args = cmdline

        _log_cmdline(self.cmdline_short_file, cmdline)
        _log_cmdline(self.cmdline_long_file, args)

        if script_file is not None:
            _log_cmdline(script_file, args)

    def calc_tmp_usage_obs(self, num_frames, prog):
        if prog not in TMP_OBS_PROGS:
            return 0

        return num_frames * self.num_track_groups * SIZEOF_FRAME_TMP

    def calc_tmp_usage(self, num_frames, prog):
        return self.calc_tmp_usage_obs(num_frames, prog) + TMP_USAGE_BASE

    def queue_task(self, prog, kwargs, job_name, num_frames,
                   output_filename=None, prefix_args=[]):
        """ Returns a restartable job object to run segway-task """
        gmtk_cmdline = prog.build_cmdline(options=kwargs)

        if prefix_args:
            # remove the command name itself from the passed arguments
            # XXX: this is ugly
            args = prefix_args + gmtk_cmdline[1:]
        else:
            args = gmtk_cmdline

        shell_job_name = extsep.join([job_name, EXT_SH])
        job_script_filename = self.job_script_dirpath() / shell_job_name

        job_script_fd = os_open(job_script_filename,
                                JOB_SCRIPT_FILE_OPEN_FLAGS,
                                JOB_SCRIPT_FILE_PERMISSIONS)
        with fdopen(job_script_fd, "w") as job_script_file:
            print("#!/usr/bin/env bash", file=job_script_file)
            # this doesn't include use of segway-wrapper, which takes the
            # memory usage as an argument, and may be run multiple times
            self.log_cmdline(gmtk_cmdline, args, job_script_file)

        if self.dry_run:
            return None

        session = self.session
        job_tmpl = session.createJobTemplate()

        job_tmpl.jobName = job_name
        job_tmpl.remoteCommand = ENV_CMD
        job_tmpl.args = [job_script_filename]

        # this is going to cause problems on heterogeneous systems
        environment = environ.copy()
        try:
            # this causes errors
            del environment["PYTHONINSPECT"]
        except KeyError:
            pass

        # Remove all post shellshock exported bash functions from the
        # environment
        job_tmpl.jobEnvironment = remove_bash_functions(environment)

        if output_filename is None:
            output_filename = self.output_dirpath() / job_name
        error_filename = self.error_dirpath() / job_name

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

    def queue_train(self, instance_index, round_index, window_index,
                    num_frames=0, **kwargs):
        """this calls Runner.queue_task()

        if num_frames is not specified, then it is set to 0, where
        everyone will share their min/max memory usage. Used for calls
        from queue_train_bundle()
        """

        kwargs["inputMasterFile"] = self.input_master_filename

        name = self.make_job_name_train(instance_index, round_index,
                                        window_index)

        segway_task_path = find_executable("segway-task")
        segway_task_verb = "run"
        output_filename = ""  # not used for training

        if window_index == NAME_BUNDLE_PLACEHOLDER:
            # Set empty window
            chrom = 0
            window_start = 0
            window_end = 0
            # Set task to the bundle train task
            task_kind = BUNDLE_TRAIN_TASK_KIND
            is_reverse = 0  # is_reverse is irrelevant for bundling

            additional_prefix_args = []
        else:
            num_frames = self.window_lens[window_index]

            # Set task to the bundle train task
            task_kind = TRAIN_TASK_KIND

            window = self.windows[window_index]
            chrom = window.chrom
            window_start = window.start
            window_end = window.end

            is_reverse = str(int(self.is_in_reversed_world(window_index)))

            track_indexes = self.world_track_indexes[window.world]
            genomedata_names = self.world_genomedata_names[window.world]

            track_indexes_text = ",".join(map(str, track_indexes))
            genomedata_names_text = ",".join(genomedata_names)

            is_semisupervised = (self.supervision_type ==
                                 SUPERVISION_SEMISUPERVISED)

            if is_semisupervised:
                supervision_coords = self.supervision_coords[window.chrom]
                supervision_labels = self.supervision_labels[window.chrom]
            else:
                supervision_coords = None
                supervision_labels = None

            virtual_evidence_coords, virtual_evidence_priors = \
                self.get_virtual_evidence_in_window(window)

            additional_prefix_args = [
               genomedata_names_text,
               self.distribution,
               track_indexes_text,
               is_semisupervised,
               supervision_coords,
               supervision_labels,
               self.virtual_evidence,
               virtual_evidence_coords,
               virtual_evidence_priors,
               self.num_segs # need number of segments to unpack VE priors later
            ]

        prefix_args = [segway_task_path,
                       segway_task_verb,
                       task_kind,
                       output_filename,  # output filename
                       chrom, window_start, window_end,
                       self.resolution,
                       is_reverse,  # end requirements for base segway-task
                       ]
        prefix_args.extend(additional_prefix_args)

        return self.queue_task(self.train_prog, kwargs, name, num_frames,
                               prefix_args=prefix_args
                               )

    def queue_train_parallel(self, input_params_filename, instance_index,
                             round_index, train_windows, **kwargs):
        kwargs["cppCommandOptions"] = \
            self.make_cpp_options(input_params_filename)

        res = RestartableJobDict(self.session, self.job_log_file)

        make_acc_filename_custom = partial(self.make_acc_filename,
                                           instance_index)

        for window_index, window_len in train_windows:
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
                           instance_index, round_index, train_windows, **kwargs
                           ):
        """bundle step: take parallel accumulators and combine them
        """
        acc_filename = self.make_acc_filename(instance_index,
                                              GMTK_INDEX_PLACEHOLDER)

        cpp_options = self.make_cpp_options(input_params_filename,
                                            output_params_filename)

        acc_range = ",".join(str(train_window[0]) for train_window in
                             train_windows)

        kwargs = dict(outputMasterFile=self.output_master_filename,
                      cppCommandOptions=cpp_options,
                      trrng="nil",
                      loadAccRange=acc_range,
                      loadAccFile=acc_filename,
                      **kwargs)

        restartable_job = self.queue_train(instance_index, round_index,
                                           NAME_BUNDLE_PLACEHOLDER, **kwargs)

        res = RestartableJobDict(self.session, self.job_log_file)
        res.queue(restartable_job)

        return res

    def queue_save_validation_set_window(self, validation_window,
                                           num_frames=0, **kwargs):
        segway_task_path = find_executable("segway-task")
        segway_task_verb = SAVE_WINDOW_TASK_VERB
        output_filename = ""

        chrom = validation_window.chrom
        window_start = validation_window.start
        window_end = validation_window.end
        name = "create_validation_set_%s_%s_%s" % (
            chrom, window_start, window_end)

        task_kind = SAVE_WINDOW_TASK_KIND

        track_indexes = self.world_track_indexes[validation_window.world]
        genomedata_names = self.world_genomedata_names[validation_window.world]

        track_indexes_text = ",".join(map(str, track_indexes))
        genomedata_names_text = ",".join(genomedata_names)

        validation_window_index = \
            self.validation_windows.index(validation_window)
        float_filepath = \
            self.validation_float_filepaths[validation_window_index]
        int_filepath = self.validation_int_filepaths[validation_window_index]

        with open(self.validation_int_filelistpath, "a") as int_filelist_fd:
            print(int_filepath, file=int_filelist_fd)

        with open(self.validation_float_filelistpath, "a") as float_filelist_fd:
            print(float_filepath, file=float_filelist_fd)

        if self.reverse_worlds:
            raise NotImplementedError("Running Segway with both validation "
                "and reverse world options simultaneously is currently "
                "not supported")
        else:
            is_reverse = 0

        prefix_args = [segway_task_path,
                       segway_task_verb,
                       task_kind,
                       output_filename,  # output filename
                       chrom, window_start, window_end,
                       self.resolution,
                       is_reverse,  # end requirements for base segway-task
                       genomedata_names_text,
                       float_filepath,
                       int_filepath,
                       self.distribution,
                       track_indexes_text,
                       self.validation_windows
                       ]

        return self.queue_task(self.train_prog, kwargs, name, num_frames,
                               prefix_args=prefix_args
                               )

    def queue_validate(self, input_params_filename, instance_index,
                       round_index, num_frames=0, **kwargs):
        """take params output from bundling and validates on validation set
        """

        cpp_options = self.make_cpp_options(input_params_filename)

        kwargs["inputMasterFile"] = self.input_master_filename

        name = self.make_job_name_validate(instance_index, round_index)

        segway_task_path = find_executable("segway-task")
        segway_task_verb = VALIDATE_TASK_VERB

        chrom = 0
        window_start = 0
        window_end = 0

        task_kind = VALIDATE_TASK_KIND

        if self.reverse_worlds:
            raise NotImplementedError
        else:
            is_reverse = 0

        prefix_args = [segway_task_path,
                       segway_task_verb,
                       task_kind,
                       self.validation_output_filename,
                       chrom, window_start, window_end,
                       self.resolution,
                       is_reverse  # end requirements for base segway-task
                       ]

        kwargs = dict(cppCommandOptions=cpp_options,
                      probE=True, cliqueTableNormalize=0.0, **kwargs)

        kwargs["of1"] = self.validation_float_filelistpath
        kwargs["of2"] = self.validation_int_filelistpath

        # delete gmtkEMtrain arguments that are not required for validation
        # using gmtkJT
        del kwargs["lldp"]
        del kwargs["maxEmIters"]
        del kwargs["objsNotToTrain"]

        restartable_job = self.queue_task(self.validate_prog,
                                          kwargs, name, num_frames,
                                          prefix_args=prefix_args
                                          )

        res = RestartableJobDict(self.session, self.job_log_file)
        res.queue(restartable_job)
        return res

    def get_posterior_clique_print_ranges(self):
        res = {}

        for clique, clique_index in viewitems(self.posterior_clique_indices):
            range_str = "%d:%d" % (clique_index, clique_index)
            res[clique + "CliquePrintRange"] = range_str

        return res

    def set_triangulation_filename(self, num_segs=None, num_subsegs=None):
        if num_segs is None:
            num_segs = self.num_segs

        if num_subsegs is None:
            num_subsegs = self.num_subsegs

        if (self.triangulation_filename_is_new or
           not self.triangulation_filename):
            self.triangulation_filename_is_new = True

            structure_filebasename = Path(self.structure_filename).name
            triangulation_filebasename = \
                extjoin(structure_filebasename, str(num_segs),
                        str(num_subsegs), EXT_TRIFILE)

            self.triangulation_filename = (self.triangulation_dirpath /
                                           triangulation_filebasename)

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
        # XXX: repetitive with queue_task
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

        if self.minibatch_fraction == MINIBATCH_DEFAULT:
            train_windows = list(self.window_lens_sorted())
        else:
            train_windows_all = list(self.window_lens_sorted())
            num_train_windows = len(train_windows_all)

            train_windows = []
            cur_bases = 0

            if num_train_windows >= MAX_GMTK_WINDOW_COUNT:
                # Workaround a GMTK bug
                # (https://gmtk-trac.bitnamiapp.com/trac/gmtk/ticket/588)
                # that raises an error if the last number in a list given to
                # -loadAccRange is greater than 9999 by always guaranteeing
                # the last number is less than MAX_GMTK_WINDOW_COUNT

                # sort train windows
                train_windows_all.sort()
                # choose all train windows with index less than
                # MAX_GMTK_WINDOW_COUNT
                valid_train_windows = train_windows_all[
                                      0:MAX_GMTK_WINDOW_COUNT
                                      ]
                # choose one train window randomly.
                # uniform chooses from the half-open interval [a, b)
                # so since window numbering begins at 0, choose between
                # [0, MAX_GMTK_WINDOW_COUNT)
                valid_gmtk_window_index = int(
                    self.random_state.uniform(0, MAX_GMTK_WINDOW_COUNT)
                    )
                # obtain the train window corresponding to the chosen
                # window index
                valid_train_window = \
                    valid_train_windows[valid_gmtk_window_index]

                # remove the chosen window from the list of windows,
                # so that the window does not get chosen again
                train_windows_all.remove(valid_train_window)

                # redefine the number of available train windows to be
                # the previous number minus 1
                num_train_windows = len(train_windows_all)

                # add the chosen window to the final list of train windows
                train_windows.append(valid_train_window)

                # start with the size of the chosen window
                cur_bases = train_windows[
                            TRAIN_WINDOWS_WINDOW_NUM_INDEX][
                            TRAIN_WINDOWS_BASES_INDEX]

            train_window_indices_shuffled = \
                self.random_state.choice(list(range(num_train_windows)),
                                         num_train_windows, replace=False)

            total_bases = sum(
                train_window[1] for train_window in train_windows_all
            )

            for train_window_index in train_window_indices_shuffled:
                if (float(cur_bases) / total_bases) < self.minibatch_fraction:
                    train_windows.append(train_windows_all[train_window_index])
                    cur_bases += train_windows_all[train_window_index][1]
                else:
                    break

            if num_train_windows >= MAX_GMTK_WINDOW_COUNT - 1:
                # Regarding the same GMTK bug as above, sort in reverse
                # to ensure that the last number in the list given to
                # -loadAccRange is our chosen window (<10000)
                train_windows.sort(reverse=True)

        restartable_jobs = \
            self.queue_train_parallel(last_params_filename, instance_index,
                                      round_index, train_windows, **kwargs)
        restartable_jobs.wait()

        restartable_jobs = \
            self.queue_train_bundle(last_params_filename, curr_params_filename,
                                    instance_index, round_index, train_windows,
                                    llStoreFile=self.log_likelihood_filename,
                                    **kwargs)
        restartable_jobs.wait()

        if self.validate:
            restartable_jobs = \
                self.queue_validate(curr_params_filename, instance_index,
                                    round_index, **kwargs)
            restartable_jobs.wait()

        self.last_params_filename = curr_params_filename

    def run_train_instance(self):
        instance_index = self.instance_index

        # If a random number generator seed exists
        if self.random_seed:
            # Create a new random number generator for this instance based on
            # its own index, so that each is different and consistent
            self.random_seed += instance_index
            self.random_state = RandomState(self.random_seed)

        self.set_triangulation_filename()

        # make new files if there is more than one instance
        new = self.instance_make_new_params

        self.set_log_likelihood_filenames(instance_index, new)
        self.set_params_filename(instance_index, new)
        if self.validate:
            self.set_validation_likelihood_filenames(instance_index, new)

        # get previous (or initial) values
        last_log_likelihood, log_likelihood, round_index, \
            validation_likelihood, best_validation_likelihood = \
            self.make_instance_initial_results()

        if round_index == 0:
            # if round > 0, this is set by self.recover_train_instance()
            self.save_input_master(instance_index)

        kwargs = dict(objsNotToTrain=self.dont_train_filename,
                      maxEmIters=1,
                      lldp=LOG_LIKELIHOOD_DIFF_FRAC * 100.0,
                      triFile=self.triangulation_filename,
                      **self.make_gmtk_kwargs())

        # var floor was specified
        if self.var_floor:
            kwargs["varFloor"] = self.var_floor

        # if using mixture models and variance floor not specified,
        # use default variance floor value to ensure convergence
        if ((self.num_mix_components != NUM_GAUSSIAN_MIX_COMPONENTS_DEFAULT)
            and not self.var_floor):
            self.var_floor = VAR_FLOOR_GMM_DEFAULT
            kwargs["varFloor"] = self.var_floor

        if self.dry_run:
            self.run_train_round(self.instance_index, round_index, **kwargs)
            return TrainInstanceResults()

        return self.progress_train_instance(last_log_likelihood,
                                            log_likelihood,
                                            round_index,
                                            validation_likelihood,
                                            best_validation_likelihood,
                                            kwargs)

    def progress_train_instance(self, last_log_likelihood, log_likelihood,
                                round_index, validation_likelihood,
                                best_validation_likelihood, kwargs):

        if (self.validate and
            self.recover_dirname):
            # recover last set of best results
            result = TrainInstanceResults(log_likelihood=log_likelihood,
                                   num_segs=self.num_segs,
                                   validation_likelihood=validation_likelihood,
                                   input_master_filename=
                                       self.input_master_filename,
                                   params_filename=self.best_params_filename,
                                   log_likelihood_filename=
                                       self.log_likelihood_filename,
                                   validation_output_filename=
                                       self.validation_output_winner_filename,
                                   validation_sum_filename=
                                       self.validation_sum_winner_filename)

        while (round_index < self.max_em_iters and
               ((self.minibatch_fraction != MINIBATCH_DEFAULT) or
                is_training_progressing(last_log_likelihood, log_likelihood))):
            self.run_train_round(self.instance_index, round_index, **kwargs)

            last_log_likelihood = log_likelihood
            log_likelihood = self.load_log_likelihood()

            if self.validate:
                # load gmtkJT output and find the validation set's likelihood
                validation_window_indices, validation_window_likelihoods = \
                    self.load_validation_log_likelihood()

                full_validation_output = \
                    self.get_full_validation_output(validation_window_indices,
                                                    validation_window_likelihoods)

                # log full set of validation likelihoods for this round
                with open(self.validation_output_tab_filename, "a") as logfile:
                    logfile.writelines(repr(full_validation_output) + "\n")

                validation_likelihood = \
                    self.calculate_validation_log_likelihood(
                        validation_window_likelihoods)

                # log latest validation set likelihood for this instance
                with open(self.validation_sum_filename, "w") as logfile:
                    logfile.write(str(validation_likelihood))

                # log final validation set likelihood for this round
                with open(self.validation_sum_tab_filename, "a") as logfile:
                    logfile.write(str(validation_likelihood) + "\n")

                # if the new validation likelihood is better than our previous
                # best, choose it as our current winner and update instance's
                # winner files winning set of results to be passed
                if validation_likelihood > best_validation_likelihood:
                    copy2(self.validation_output_filename,
                          self.validation_output_winner_filename)

                    copy2(self.validation_sum_filename,
                          self.validation_sum_winner_filename)

                    result = TrainInstanceResults(log_likelihood=log_likelihood,
                                   num_segs=self.num_segs,
                                   validation_likelihood=validation_likelihood,
                                   input_master_filename=
                                       self.input_master_filename,
                                   params_filename=self.best_params_filename,
                                   log_likelihood_filename=
                                       self.log_likelihood_filename,
                                   validation_output_filename=
                                       self.validation_output_winner_filename,
                                   validation_sum_filename=
                                       self.validation_sum_winner_filename)
                    best_validation_likelihood = validation_likelihood

            round_index += 1

        if not self.validate:
            result = TrainInstanceResults(log_likelihood=log_likelihood,
                                   num_segs=self.num_segs,
                                   validation_likelihood=validation_likelihood,
                                   input_master_filename=
                                       self.input_master_filename,
                                   params_filename=self.last_params_filename,
                                   log_likelihood_filename=
                                       self.log_likelihood_filename,
                                   validation_output_filename=
                                       self.validation_output_winner_filename,
                                   validation_sum_filename=
                                       self.validation_sum_winner_filename)

        # log_likelihood, num_segs and a list of src_filenames to save
        return result

    def save_tab_file(self, values, attrs, tab_filename):
        """
        Creates a .tab file to store specific variables between steps, such as
        train results between run and finish, as well as options between
        steps and tasks, such as between train and identify.
        """

        filename = self.make_filename(tab_filename)

        with open(filename, "w") as tab_file:
            writer = ListWriter(tab_file, delimiter=DELIMITER_BED)
            writer.writerow(TRAIN_FIELDNAMES)

            for name, object_type in sorted(viewitems(attrs)):
                value = getattr(values, name)
                if isinstance(object_type, list):
                    for item in value:
                        writer.writerow([name, item])
                else:
                    writer.writerow([name, value])

    def load_train_options(self, train_dirname):
        """
        load options from training
        """
        filename = Path(train_dirname) / TRAIN_FILEBASENAME

        with open(filename) as tabfile:
            reader = DictReader(tabfile)

            for row in reader:
                name = row["name"]
                value = row["value"]

                # Check if the option is used by identify as well, then ignore
                is_identify_option = name in IDENTIFY_OPTION_TYPES.keys() and \
                    (self.identify or self.posterior)
                if value and not is_identify_option:
                    row_type = TRAIN_OPTION_TYPES[name]
                    if isinstance(row_type, list):
                        assert len(row_type) == 1
                        item_type = row_type[0]
                        getattr(self, name).append(item_type(value))
                    else:
                        setattr(self, name, row_type(value))

        # If we are running identify/posterior and the params filename was
        # specified during train, save it in params_filenames as well.
        if self.params_filename is not None and \
          not self.train:
            self.params_filenames = [self.params_filename]

    def load_train_results(self):
        """
        Loads the training results in train-finish saved in 
        SUBDIRNAME_INTERMEDIATE. Stores as a list of TrainInstanceResults in
        instance_params to select best in finish.
        """

        pattern = extjoin(PREFIX_TRAIN_RESULTS, "*", EXT_TAB)
        instance_params = []
        for filename in Path(self.intermediate_dirpath).files(pattern):
            # Set variables to none to be overwritten by the tab file contents
            results = TrainInstanceResults()
            results.load_results(filename)
            instance_params.append(results)

        return instance_params

    def load_identify_options(self, identify_dir_name):
        """
        load options set in identify-init, stored in the identify.tab file
        in the main directory.
        """

        filename = Path(identify_dir_name) / IDENTIFY_FILEBASENAME

        with open(filename) as tabfile:
            reader = DictReader(tabfile)

            for row in reader:
                name = row["name"]
                value = row["value"]

                row_type = IDENTIFY_OPTION_TYPES[name]
                # Checks if this option was of type list, sets each element to
                # correct type then
                if isinstance(row_type, list):
                    item_type = row_type[0]
                    getattr(self, name).append(item_type(value))
                else:
                    # Overwrite the empty string to save as None
                    if value == "":
                        setattr(self, name, None)
                    else:
                        setattr(self, name, row_type(value))

    def init_shared(self):
        """
        Perform initialization shared by all of train, posterior and identify.
        """

        self.make_subdirs(SUBDIRNAMES_EITHER)

        self.save_gmtk_input()
        self.save_window_list()
        self.run_triangulate()

    def init_train(self):
        """
        return value: dst_filenames
        """

        assert self.num_instances >= 1

        self.init_shared()
        self.make_subdirs(SUBDIRNAMES_TRAIN)

        # save the destination file for input_master as we will be
        # generating new input masters for each start

        # must be before file creation. Otherwise
        # input_master_filename_is_new will be wrong

        input_master_filename, input_master_filename_is_new = \
            InputMasterSaver(self)(self.input_master_filename,
                                   self.params_dirpath, self.clobber)
        self.input_master_filename = input_master_filename

        # save file locations to tab-delimited file
        self.save_tab_file(self, TRAIN_OPTION_TYPES, TRAIN_FILEBASENAME)

        # If a single instance is being run, create a single input.master
        # Else create an input master for each instance
        if not self.instance_make_new_params:
            self.save_input_master()
        else:
            for index in range(self.num_instances):
                # If a random number generator seed exists
                if self.random_seed:
                    # Create a new random number generator for this instance based on its
                    # own index
                    instance_random_seed = self.random_seed + index
                    self.random_state = RandomState(instance_random_seed)
                self.save_input_master(index, True)


        if not input_master_filename_is_new:
            # do not overwrite existing file
            input_master_filename = None

    def get_thread_run_func(self):
        if len(self.num_segs_range) > 1 or self.num_instances > 1:
            return self.run_train_multithread
        else:
            return self.run_train_singlethread

    def finish_train(self, instance_params):
        """
        Finds and then copies the best training round and instance to the
        general files such as params.params, input.master, etc.
        """

        if self.dry_run:
            return

        # Copy the param from the best round of each training instance
        for instance in instance_params:
            # Remove iteration number from end of winning param filename
            winning_instance_param_filename = \
                instance.params_filename.rpartition(".")[0]
            copy2(instance.params_filename, winning_instance_param_filename)

        # finds max log_likelihood for validation or regular
        if self.validate:
            max_params = max(instance_params,
                             key=attrgetter("validation_likelihood"))
        else:
            max_params = max(instance_params, key=attrgetter("log_likelihood"))

        src_filenames = max_params.get_filenames(self.validate)

        if None in src_filenames.values():
            raise RuntimeError("All training instances failed")

        best_params_filename = max_params.params_filename

        for filename in src_filenames:
            self.copy_results(filename, src_filenames[filename])

        # always overwrite params.params
        copy2(best_params_filename, self.make_params_filename())

    def save_validation_set(self):
        with Session() as session:
            self.session = session
            self.instance_index = "validation"
            with self.open_job_log_file() as self.job_log_file:
                res = RestartableJobDict(self.session, self.job_log_file)
                for validation_window in self.validation_windows:
                    restartable_job = \
                        self.queue_save_validation_set_window(validation_window)
                    res.queue(restartable_job)
                res.wait()
        self.session = None
        self.instance_index = None

    def run_train(self):

        if self.train.init:
            self.init_train()
        else:
            # Reload filenames
            self.save_gmtk_input()

        if self.train.run:
            run_train_func = self.get_thread_run_func()

            # this is where the actual training takes place
            instance_params = run_train_func(self.num_segs_range)
            # write results from each instance to their own file
            for index, instance_param in enumerate(instance_params):
                result_filename = make_default_filename \
                    (TRAIN_RESULTS_TMPL, SUBDIRNAME_INTERMEDIATE, index)
                self.save_tab_file(instance_param, TRAIN_RESULT_TYPES,
                                  result_filename)

        elif self.train.finish:
            instance_params = self.load_train_results()

        if self.train.finish:
            self.finish_train(instance_params)

    def run_train_singlethread(self, num_segs_range):
        # having a single-threaded version makes debugging much easier
        with Session() as session:
            self.session = session
            self.instance_index = 0
            with self.open_job_log_file() as self.job_log_file:
                res = [self.run_train_instance()]

        self.session = None

        return res

    def run_train_multithread(self, num_segs_range):
        seg_instance_indexes = range(self.num_instances)
        enumerator = enumerate(product(num_segs_range, seg_instance_indexes))

        # ensure memoization before threading
        self.triangulation_dirpath
        self.jt_info_filename
        self.include_coords
        self.exclude_coords
        self.validation_coords
        self.minibatch_fraction
        self.card_seg_countdown
        self.obs_dirpath
        self.float_filelistpath
        self.int_filelistpath
        self.validation_float_filelistpath
        self.validation_int_filelistpath
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
        self.validate_prog
        self.supervision_label_range_size

        threads = []
        with Session() as session:
            for instance_index, instance_features in enumerator:
                num_seg, seg_instance_index = instance_features

                thread = TrainThread(self, session, instance_index, num_seg)
                thread.start()
                threads.append(thread)

            # list of tuples(log_likelihood, input_master_filename,
            #                params_filename)
            instance_params = []
            for thread in threads:
                while thread.is_alive():
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

        return instance_params

    def recover_filename(self, resource):
        instance_index = self.instance_index

        # only want "input.master" not "input.0.master" if there is
        # only one instance
        if (not self.instance_make_new_params and
           resource == InputMasterSaver.resource_name):
            instance_index = None

        old_filename = make_default_filename(resource,
                                             self.recover_params_dirpath,
                                             instance_index)

        new_filename = make_default_filename(resource, self.params_dirpath,
                                             instance_index)
        Path(old_filename).copy2(new_filename)
        return new_filename

    def recover_train_instance(self, last_log_likelihood, log_likelihood,
                               validation_likelihood,
                               best_validation_likelihood):
        instance_index = self.instance_index
        recover_dirname = self.recover_dirname

        recover_log_likelihood_tab_filepath = \
            Path(self.make_log_likelihood_tab_filename(instance_index,
                                                       recover_dirname))

        if not recover_log_likelihood_tab_filepath.isfile():
            # If the likelihood tab file does not exist in this case, then it
            # means the recover directory was running a single instance which
            # would not generate any tab file with an instance labeled suffix.
            # Read the tab file that does not have a labeled number suffix.
            recover_log_likelihood_tab_filepath = \
                Path(self.make_log_likelihood_tab_filename(None,
                                                           recover_dirname))

        if not self.recover_round:
            # If we are recovering a previous training instance rather than
            # a round we need to copy the files from the old training directory
            # into our current one. If just using train-run-round, these will
            # already be present.
            self.input_master_filename = \
                self.recover_filename(InputMasterSaver.resource_name)
            log_likelihood_tab_filename = self.log_likelihood_tab_filename
            recover_log_likelihood_tab_filepath.copy2(
                log_likelihood_tab_filename)

        if self.validate:
            # recover validation sum tabfile
            recover_validation_sum_tab_filepath = \
                Path(self.make_validation_sum_tab_filename(instance_index,
                                                           recover_dirname))
            if not recover_validation_sum_tab_filepath.isfile():
                recover_validation_sum_tab_filepath = \
                    Path(self.make_validation_sum_tab_filename(None,
                                                           recover_dirname))

            # recover validation output tabfile
            recover_validation_output_tab_filepath = \
                Path(self.make_validation_output_tab_filename(instance_index,
                                                           recover_dirname))
            if not recover_validation_output_tab_filepath.isfile():
                recover_validation_output_tab_filepath = \
                    Path(self.make_validation_output_tab_filename(None,
                                                           recover_dirname))

            # recover last validation sum
            recover_validation_sum_filename = \
                Path(self.make_filename(PREFIX_VALIDATION_SUM,
                                       instance_index, EXT_LIKELIHOOD,
                                       subdirname=SUBDIRNAME_LIKELIHOOD,
                                       dirname=recover_dirname))
            if not recover_validation_sum_filename.isfile():
                recover_validation_sum_filename = \
                    Path(self.make_filename(PREFIX_VALIDATION_SUM,
                                            None, EXT_LIKELIHOOD,
                                            subdirname=SUBDIRNAME_LIKELIHOOD,
                                            dirname=recover_dirname))

            # recover last validation output
            recover_validation_output_filename = \
                Path(self.make_filename(PREFIX_VALIDATION_OUTPUT,
                                        instance_index, EXT_LIKELIHOOD,
                                        subdirname=SUBDIRNAME_LIKELIHOOD,
                                        dirname=recover_dirname))
            if not recover_validation_output_filename.isfile():
                recover_validation_output_filename = \
                    Path(self.make_filename(PREFIX_VALIDATION_OUTPUT,
                                            None, EXT_LIKELIHOOD,
                                            subdirname=SUBDIRNAME_LIKELIHOOD,
                                            dirname=recover_dirname))

            # recover validation sum winner
            recover_validation_sum_winner_filename = \
                Path(self.make_filename(PREFIX_VALIDATION_SUM_WINNER,
                                        instance_index, EXT_LIKELIHOOD,
                                        subdirname=SUBDIRNAME_LIKELIHOOD,
                                        dirname=recover_dirname))
            if not recover_validation_sum_winner_filename.isfile():
                recover_validation_sum_winner_filename = \
                    Path(self.make_filename(PREFIX_VALIDATION_SUM_WINNER,
                                            None, EXT_LIKELIHOOD,
                                            subdirname=SUBDIRNAME_LIKELIHOOD,
                                            dirname=recover_dirname))

            # recover validation output winner
            recover_validation_output_winner_filename = \
                Path(self.make_filename(PREFIX_VALIDATION_OUTPUT_WINNER,
                                        instance_index, EXT_LIKELIHOOD,
                                        subdirname=SUBDIRNAME_LIKELIHOOD,
                                        dirname=recover_dirname))
            if not recover_validation_output_winner_filename.isfile():
                recover_validation_output_winner_filename = \
                    Path(self.make_filename(PREFIX_VALIDATION_OUTPUT_WINNER,
                                            None, EXT_LIKELIHOOD,
                                            subdirname=SUBDIRNAME_LIKELIHOOD,
                                            dirname=recover_dirname))
            if not self.recover_round:
                copy2(recover_validation_sum_tab_filepath,
                      self.validation_sum_tab_filename)
                copy2(recover_validation_output_tab_filepath,
                      self.validation_output_tab_filename)
                copy2(recover_validation_sum_filename,
                      self.validation_sum_filename)
                copy2(recover_validation_output_filename,
                      self.validation_output_filename)
                copy2(recover_validation_sum_winner_filename,
                      self.validation_sum_winner_filename)
                copy2(recover_validation_output_winner_filename,
                      self.validation_output_winner_filename)

        # Check if a round had been finished, and produced a likelihood
        if recover_log_likelihood_tab_filepath.isfile():
            with open(recover_log_likelihood_tab_filepath) \
                    as log_likelihood_tab_file:
                log_likelihoods = [float(line.rstrip())
                                   for line in log_likelihood_tab_file.readlines()]
        else:
            # Otherwise, set likelihoods to empty
            log_likelihoods = []

        final_round_index = len(log_likelihoods)
        if final_round_index == 0:
            # This is the case where train-run-round was used for the first
            # round sets the log_likelihood temporarily as it would be
            # otherwise
            log_likelihood = -inf
        if final_round_index > 0:
            log_likelihood = log_likelihoods[-1]
        if final_round_index > 1:
            last_log_likelihood = log_likelihoods[-2]

        if self.validate:
            with open(recover_validation_sum_tab_filepath) \
                as validation_sum_tab_file:
                validation_sums = [float(line.rstrip())
                                   for line in validation_sum_tab_file.readlines()]
            validation_likelihood = validation_sums[-1]
            best_validation_likelihood = max(validation_sums)
            best_validation_index = validation_sums.index(
                best_validation_likelihood)


        old_params_filename = self.make_params_filename(instance_index,
                                                        recover_dirname)
        new_params_filename = self.params_filename
        for round_index in range(final_round_index):
            old_curr_params_filename = extjoin(old_params_filename,
                                               str(round_index))
            new_curr_params_filename = extjoin(new_params_filename,
                                               str(round_index))

            if not self.recover_round:
                Path(old_curr_params_filename).copy2(new_curr_params_filename)
            if self.validate:
                if round_index == best_validation_index:
                    self.best_params_filename = new_curr_params_filename

        if final_round_index:
            self.last_params_filename = new_curr_params_filename
        if self.recover_round:
            self.max_em_iters = final_round_index + 1

        return last_log_likelihood, log_likelihood, final_round_index, \
            validation_likelihood, best_validation_likelihood

    def make_instance_initial_results(self):
        """
        returns last_log_likelihood, log_likelihood, round_index
        -inf, -inf, 0 if there is no recovery--this is also used to set initial
        values
        """
        # initial values:
        last_log_likelihood = -inf
        log_likelihood = -inf
        final_round_index = 0
        validation_likelihood = -inf
        best_validation_likelihood = -inf

        if self.recover_dirpath:
            return self.recover_train_instance(last_log_likelihood,
                                               log_likelihood,
                                               validation_likelihood,
                                               best_validation_likelihood)

        return last_log_likelihood, log_likelihood, final_round_index, \
            validation_likelihood, best_validation_likelihood

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
        except IOError as err:
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
        Path(recover_filename).copy2(self.viterbi_filenames[window_index])

        print("window", window_index, "already complete", file=sys.stderr)

        return True

    def queue_identify(self, restartable_jobs, window_index, prefix_job_name,
                       prog, kwargs, output_filenames):
        prog = self.prog_factory(prog)
        job_name = self.make_job_name_identify(prefix_job_name, window_index)
        output_filename = output_filenames[window_index]

        kwargs = self.get_identify_kwargs(window_index, kwargs)

        if prog == VITERBI_PROG:
            kind = ANNOTATE_TASK_KIND
        else:
            kind = POSTERIOR_TASK_KIND

        # "0" or "1"
        is_reverse = str(int(self.is_in_reversed_world(window_index)))

        window = self.windows[window_index]

        track_indexes = self.world_track_indexes[window.world]
        genomedata_names = self.world_genomedata_names[window.world]

        track_indexes_text = ",".join(map(str, track_indexes))
        genomedata_names_text = ",".join(genomedata_names)

        virtual_evidence_coords, virtual_evidence_priors = \
            self.get_virtual_evidence_in_window(window)

        # Prefix args all get mapped with "str" function!
        prefix_args = [find_executable("segway-task"), "run", kind,
                       output_filename, window.chrom,
                       window.start, window.end, self.resolution, is_reverse,
                       self.num_segs, self.num_subsegs, self.output_label,
                       genomedata_names_text,
                       self.distribution, track_indexes_text,
                       self.virtual_evidence,
                       virtual_evidence_coords,
                       virtual_evidence_priors,
                       self.num_mix_components,]

        output_filename = None

        num_frames = self.window_lens[window_index]

        restartable_job = self.queue_task(prog, kwargs, job_name,
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

        if self.is_in_reversed_world(window_index):
            res["gpr"] = REVERSE_GPR

        res.update(extra_kwargs)

        return res

    def is_in_reversed_world(self, window_index):
        return self.windows[window_index].world in self.reverse_worlds

    def save_identify_posterior(self):
        for world in range(self.num_worlds):
            if self.identify.finish:
                IdentifySaver(self)(world)

            if self.posterior.finish:
                PosteriorSaver(self)(world)

    def run_identify_posterior(self):
        self.instance_index = "identify"

        if self.identify.init or self.posterior.init:
            self.init_shared()
            self.save_tab_file(self, IDENTIFY_OPTION_TYPES,
                               IDENTIFY_FILEBASENAME)
            self.make_subdir(SUBDIRNAME_VITERBI)
            self.make_subdir(SUBDIRNAME_POSTERIOR)
        else:
            self.set_triangulation_filename()
            self.load_identify_options(self.work_dirname)
            self.save_gmtk_input()

        # if output_label == "subseg" or "full", need to catch
        # superlabel and sublabel output from gmtk
        if self.output_label != "seg":
            VITERBI_REGEX_FILTER = "^(seg|subseg)$"
        else:
            VITERBI_REGEX_FILTER = "^seg$"

        # -: standard output, processed by segway-task
        kwargs = {"identify":
                  dict(triFile=self.triangulation_filename,
                       pVitRegexFilter=VITERBI_REGEX_FILTER,
                       cVitRegexFilter=VITERBI_REGEX_FILTER,
                       eVitRegexFilter=VITERBI_REGEX_FILTER,
                       vitCaseSensitiveRegexFilter=True, mVitValsFile="-"),
                  "posterior":
                  dict(triFile=self.posterior_triangulation_filename,
                       jtFile=self.posterior_jt_info_filename,
                       doDistributeEvidence=True,
                       **self.get_posterior_clique_print_ranges())}

        tasks = []
        filenames = {}
        if self.identify.run:
            tasks.append("identify")
            filenames.update(dict(identify=self.viterbi_filenames))
        if self.posterior.run:
            tasks.append("posterior")
            filenames.update(dict(posterior=self.posterior_filenames))

        with Session() as session:
            self.session = session
            with self.open_job_log_file() as self.job_log_file:
                restartable_jobs = RestartableJobDict(session,
                                                      self.job_log_file)

                for window_index, window_len in self.window_lens_sorted():
                    for task in tasks:
                        if (task == "identify" and
                        self.recover_viterbi_window(window_index)):
                            # XXX: should be able to recover posterior also
                            continue

                        self.queue_identify(restartable_jobs, window_index,
                                            PREFIX_JOB_NAMES[task], PROGS[task],
                                            kwargs[task], filenames[task])

                if self.dry_run:
                    return

                restartable_jobs.wait()

        self.save_identify_posterior()

    def make_script_filename(self, prefix):
        return self.make_filename(prefix, EXT_SH, subdirname=SUBDIRNAME_LOG)

    def make_run_msg(self):
        now = datetime.now()
        pkg_desc = working_set.find(Requirement.parse(__package__))
        run_msg = "## %s run %s at %s" % (pkg_desc, self.uuid, now)

        cmdline_top_filename = self.make_script_filename(PREFIX_CMDLINE_TOP)

        with open(cmdline_top_filename, "a") as cmdline_top_file:
            print(run_msg, file=cmdline_top_file)
            print(file=cmdline_top_file)
            print("cd", maybe_quote_arg(Path.getcwd()), file=cmdline_top_file)
            print(cmdline2text(), file=cmdline_top_file)

        return run_msg

    def run(self):
        """
        main run, after dirname is specified

        this is exposed so that it can be overriden in a subclass

        opens log files, saves parameters, and calls main function
        run_train() or run_identify_posterior()
        """
        # XXXopt: use binary I/O to gmtk rather than ascii for parameters

        # start log files, if they weren't already created by an ealier subtask
        if not Path(self.work_dirpath / SUBDIRNAME_LOG).exists():
            self.make_subdir(SUBDIRNAME_LOG)
        run_msg = self.make_run_msg()

        cmdline_short_filename = \
            self.make_script_filename(PREFIX_CMDLINE_SHORT)
        cmdline_long_filename = self.make_script_filename(PREFIX_CMDLINE_LONG)

        # Check if the genomedata archives are compatable with each other
        if not self.check_genomedata_archives():
            raise ValueError(
                "Genomedata archives do not have matching chromosome starts "
                "and ends"
            )

        with open(cmdline_short_filename, "a") as self.cmdline_short_file:
            with open(cmdline_long_filename, "a") as self.cmdline_long_file:
                print(run_msg, file=self.cmdline_short_file)
                print(run_msg, file=self.cmdline_long_file)

                if (self.validate and
                    self.train.init):
                    self.save_validation_set()

                if self.train:
                    self.run_train()

                if self.identify or self.posterior:
                    if self.posterior:
                        if self.recover_dirname:
                            raise NotImplementedError(
                                "Recovery is not yet supported for the "
                                "posterior task"
                            )
                        if self.num_worlds != 1:
                            raise NotImplementedError(
                                "Tied tracks are not yet supported for "
                                "the posterior task"
                            )

                    self.run_identify_posterior()

    def __call__(self, *args, **kwargs):
        # XXX: register atexit for cleanup_resources

        work_dirname = self.work_dirname
        if not Path(work_dirname).isdir() or self.clobber:
            self.make_dir(work_dirname, self.clobber)

        self.run(*args, **kwargs)


def parse_options(argv):
    from argparse import ArgumentParser, FileType

    version = "%(prog)s {}".format(__version__)
    description = """
    Segmentation and automated genome annotation."""

    usage = "segway [global_args] COMMAND [args]..."

    citation = """
    Citation: Hoffman MM, Buske OJ, Wang J, Weng Z, Bilmes J, Noble WS.  2012.
    Unsupervised pattern discovery in human chromatin structure through genomic
    segmentation. Nat Methods 9:473-476.
    https://doi.org/10.1038/nmeth.1937"""

    parser = ArgumentParser(description=description, usage = usage,
                            epilog=citation)

    subtask_description = """
train                   create a model with learned parameters
- train-init            prepare initial models for parallel training
- train-run             train initial models to completion criterion, in parallel
-- train-run-round      train models for one round, in parallel
- train-finish          select best model and prepare for `annotate` 

annotate                label a genome using a model
- annotate-init         prepare for parallel annotation
- annotate-run          annotate the genome, in parallel
- annotate-finish       concatenate parallel annotation results

posterior               infer posterior probabilities of each label across a genome
- posterior-init        prepare for parallel posterior inference
- posterior-run         infer posterior probabilities, in parallel
- posterior-finish      concatenate parallel posterior inference results

Use `segway COMMAND --help` for help specific to command COMMAND.

    """

    tasks = parser.add_subparsers(title="list of commands", dest="task_spec",
                                  metavar=subtask_description)

    parser.add_argument("--version", action="version", version=version)

    # set global options first
    group = parser.add_argument_group("Technical variables")
    group.add_argument("-m", "--mem-usage", default=MEM_USAGE_PROGRESSION,
                       metavar="PROGRESSION",
                       help="try each float in PROGRESSION as the number "
                       "of gibibytes of memory to allocate in turn "
                       "(default %s)" % MEM_USAGE_PROGRESSION)

    group.add_argument("-S", "--split-sequences", metavar="SIZE",
                       default=MAX_SPLIT_SEQUENCE_LENGTH, type=int,
                       help="split up sequences that are larger than SIZE "
                       "bp (default %s)" % MAX_SPLIT_SEQUENCE_LENGTH)

    group.add_argument("-v", "--verbosity", type=int, default=VERBOSITY,
                       metavar="NUM",
                       help="show messages with verbosity NUM"
                       " (default %d)" % VERBOSITY)

    group.add_argument("--cluster-opt", action="append", default=[],
                       metavar="OPT",
                       help="specify an option to be passed to the "
                       "cluster manager")

    group = parser.add_argument_group("Flags")
    group.add_argument("-n", "--dry-run", action="store_true",
                       help="write all files, but do not run any executables")

    # define steps for each of the three main tasks
    train_init = tasks.add_parser("", add_help=False)
    train_run = tasks.add_parser("", add_help=False)
    train_finish = tasks.add_parser("", add_help=False)
    train_run_round = tasks.add_parser("", add_help=False)
    train_init_run = tasks.add_parser("", add_help=False)

    identify_init = tasks.add_parser("", add_help=False)
    identify_run = tasks.add_parser("", add_help=False)
    identify_finish = tasks.add_parser("", add_help=False)
    identify_init_run = tasks.add_parser("", add_help=False)

    # Common options across tasks
    class Argument(object):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
    
    include_coords_args = Argument("--include-coords",
        metavar="FILE",
        help="limit to genomic coordinates in" " FILE (default all)"
             " (Note: does not apply to" " --validation-coords)",
    )

    exclude_coords_args = Argument("--exclude-coords",
        metavar="FILE",
        help="filter out genomic coordinates in FILE (default none)",
    )

    clobber_args = Argument("-c", "--clobber",
        action="store_true",
        help="delete any preexisting files and assumes any model files"
             "specified in options as output to be overwritten",
    )

    seg_table_args = Argument("--seg-table",
        metavar="FILE",
        help="load segment hyperparameters from FILE (default none)",
    )

    virtual_evidence_args = Argument ("--virtual-evidence",
        metavar="FILE",
        help="virtual evidence with priors for labels at each position"
             " in FILE (default none)",
    )

    recover_args = Argument ("-r", "--recover",
        metavar="DIR",
        help="continue from interrupted run in DIR",
    )

    # next two groups of options belong in train-init
    # with OptionGroup(parser, "Data selection") as group:
    group = train_init.add_argument_group("Data selection (train-init)")
    group.add_argument("-t", "--track", action="append", default=[],
                       metavar="track", help="append track to list of tracks"
                       " to use (default all)")

    # TODO: Fix action "load" and use FileType instead
    group.add_argument("--tracks-from", type=FileType("r"), metavar="FILE",
                       dest="track", nargs=1, help="append tracks from"
                       " newline-delimited FILE to list of tracks"
                       " to use")

    group.add_argument(*include_coords_args.args, **include_coords_args.kwargs)

    # exclude goes after all includes
    group.add_argument(*exclude_coords_args.args, **exclude_coords_args.kwargs)

    group.add_argument("--resolution", type=int, metavar="RES",
                       help="downsample to every RES bp (default %d)" %
                       RESOLUTION)

    group.add_argument("--minibatch-fraction", type=float, metavar="FRAC",
                       default=MINIBATCH_DEFAULT,
                       help="Use a random fraction FRAC positions for each EM"
                       " iteration. Removes the likelihood stopping criterion,"
                       " so training always runs to the number of rounds"
                       " specified by --max-train-rounds.")

    group.add_argument("--validation-fraction", type=float, metavar="FRAC",
                       help="Use a random held out set of size FRAC of the"
                       " included genomic regions to validate the parameters"
                       " learned by each training round. The instance/round"
                       " with the best likelihood as validated by the holdout"
                       " set will be chosen as the winner after the specified"
                       " --max-train-rounds.")

    group.add_argument("--validation-coords", metavar="FILE",
                       help="Use genomic coordinates in FILE as a validation"
                       " set (default none)")

    group = train_init.add_argument_group("Model files (train-init)")
    group.add_argument("-i", "--input-master", metavar="FILE",
                       help="use or create input master in FILE"
                       " (default %s)" %
                       make_default_filename(InputMasterSaver.resource_name,
                                             DIRPATH_PARAMS))

    group.add_argument("-s", "--structure", metavar="FILE",
                       help="use or create structure in FILE (default %s)" %
                       make_default_filename(StructureSaver.resource_name))

    group.add_argument("-p", "--trainable-params", action="append",
                       default=[], metavar="FILE",
                       help="use or create trainable parameters in FILE"
                       " (default WORKDIR/params/params.params)")

    group.add_argument("--dont-train", metavar="FILE",
                       help="use FILE as list of parameters not to train"
                       " (default %s)" %
                       make_default_filename(RES_DONT_TRAIN, DIRPATH_AUX))

    group.add_argument(*seg_table_args.args, **seg_table_args.kwargs)

    group.add_argument("--semisupervised", metavar="FILE",
                       help="semisupervised segmentation with labels in "
                       "FILE (default none)")

    group = train_init.add_argument_group("Modeling variables (train-init)")
    group.add_argument("-D", "--distribution", choices=DISTRIBUTIONS,
                       metavar="DIST",
                       help="use DIST distribution"
                       " (default %s)" % DISTRIBUTION_DEFAULT)

    group.add_argument("--mixture-components", type=int,
                         default=NUM_GAUSSIAN_MIX_COMPONENTS_DEFAULT,
                         help="Number of Gaussian mixture "
                         "components (default %d)"
                         % NUM_GAUSSIAN_MIX_COMPONENTS_DEFAULT)

    group.add_argument("--num-instances", type=int,
                       default=NUM_INSTANCES, metavar="NUM",
                       help="run NUM training instances, randomizing start"
                       " parameters NUM times (default %d)" % NUM_INSTANCES)

    group.add_argument("-N", "--num-labels", type=str, metavar="SLICE",
                       help="make SLICE segment labels"
                       " (default %d)" % NUM_SEGS)  # will use str2slice_or_int

    group.add_argument("--num-sublabels", type=int, metavar="NUM",
                       help="make NUM segment sublabels"
                       " (default %d)" % NUM_SUBSEGS)

    group.add_argument("--ruler-scale", type=int, metavar="SCALE",
                       help="ruler marking every SCALE bp (default the"
                       " resolution multiplied by 10)")

    group.add_argument("--prior-strength", type=float, metavar="RATIO",
                       help="use RATIO times the number of data counts as"
                       " the number of pseudocounts for the segment length"
                       " prior (default %f)" % PRIOR_STRENGTH)

    group.add_argument("--segtransition-weight-scale", type=float,
                       metavar="SCALE",
                       help="exponent for segment transition probability "
                       " (default %f)" % SEGTRANSITION_WEIGHT_SCALE)

    group.add_argument("--track-weight", type=float,
                       help="exponent for all input track probabilities "
                       " (default 1)")

    group.add_argument("--virtual-evidence-weight", type=float,
                       help="exponent for virtual evidence probability "
                       " (default %f)" % VIRTUAL_EVIDENCE_WEIGHT)

    group.add_argument("--reverse-world", action="append", type=int,
                       default=[], metavar="WORLD",
                       help="reverse sequences in concatenated world WORLD"
                       " (0-based)")

    group.add_argument("--var-floor", type=float, default=None,
                         help="Sets the variance floor, meaning that if any "
                         "of the variances of a track falls below "
                         'this value, then the variance is "floored" (prohibited '
                         "from falling below the floor value). If not using a "
                         "mixture of Gaussians, default unused, else default %f"
                         % VAR_FLOOR_GMM_DEFAULT)

    group = train_init.add_argument_group("Flags (train-init)")
    group.add_argument(*clobber_args.args, **clobber_args.kwargs)

    # Train run is where the train-rounds are calculated
    group = train_run.add_argument_group("Modeling Variables (train-run)")
    group.add_argument("--max-train-rounds", type=int, metavar="NUM",
                       help="each training instance runs a maximum of NUM"
                       " rounds (default %d)" % MAX_EM_ITERS)

    # Directory would be recovered from train-run
    group = train_run.add_argument_group("Intermediate files (train-run,"
                                         " annotate-run)")
    group.add_argument("-o", "--observations", metavar="DIR",
                       help="DEPRECATED - temp files are now used and "
                       "recommended. Previously would use or create "
                       "observations in DIR (default %s)" %
                       (DIRPATH_WORK_DIR_HELP / SUBDIRNAME_OBS))

    group.add_argument(*recover_args.args, **recover_args.kwargs)

    group = train_init_run.add_argument_group("Virtual Evidence "
                                              "(train-init, train-run)")
    group.add_argument(*virtual_evidence_args.args, **virtual_evidence_args.kwargs)

    group = identify_init.add_argument_group("Flags "
                                             "(annotate-init, posterior-init)")
    group.add_argument(*clobber_args.args, **clobber_args.kwargs)

    # select coords to identify in the init step
    group = identify_init.add_argument_group("Data selection "
                                             "(annotate-init, posterior-init)")

    group.add_argument(*include_coords_args.args, **include_coords_args.kwargs)

    # exclude goes after all includes
    group.add_argument(*exclude_coords_args.args, **exclude_coords_args.kwargs)

    group.add_argument("--output-label", type=str,
                       help="in the segmentation file, for each coordinate "
                       'print only its superlabel ("seg"), only its '
                       "sublabel (\"subseg\"), or both (\"full\")"
                       "  (default %s)" % OUTPUT_LABEL)

    group.add_argument(*seg_table_args.args, **seg_table_args.kwargs)

    group = identify_init_run.add_argument_group("Virtual Evidence "
                                                 "(annotate-init, "
                                                 "posterior-init, "
                                                 "annotate-run, "
                                                 "posterior-run)")
    group.add_argument(*virtual_evidence_args.args, **virtual_evidence_args.kwargs)

    group = identify_run.add_argument_group("Intermediate files (train-run,"
                                            " annotate-run)")
    group.add_argument(*recover_args.args, **recover_args.kwargs)

    # output files are produced by identify-finish
    group = identify_finish.add_argument_group("Output files (annotate-finish)")
    group.add_argument("-b", "--bed", metavar="FILE",
                       help="create identification BED track in FILE"
                       " (default WORKDIR/%s)" % BED_FILEBASENAME)

    group.add_argument("--bigBed", metavar="FILE",
                       help="specify layered bigBed filename")

    args = tasks.add_parser("", add_help=False)
    # Positional arguments
    args.add_argument("archives", nargs="+")  # "+" for at least 1 arg
    args.add_argument("traindir", nargs=1)

    identify_args = tasks.add_parser("", add_help=False)
    identify_args.add_argument("annotatedir", nargs=1)

    tasks.add_parser("train-init", parents=[train_init, train_init_run, args],
                     usage = "segway train-init [args] GENOMEDATA TRAINDIR")
    tasks.add_parser("train-run", parents=[train_run, train_init_run, args],
                     usage = "segway train-run [args] GENOMEDATA TRAINDIR")
    tasks.add_parser("train-finish", parents=[train_finish, args],
                     usage = "segway train-finish [args] GENOMEDATA TRAINDIR")
    tasks.add_parser("train-run-round", parents=[train_run_round,
                                                 train_init_run, args],
                     usage = "segway train-run-round [args] GENOMEDATA TRAINDIR")

    tasks.add_parser("annotate-init",
                     parents=[identify_init, identify_init_run, args,
                              identify_args],
                     usage = "segway annotate-init [args] GENOMEDATA TRAINDIR ANNOTATEDIR")
    tasks.add_parser("annotate-run",
                     parents=[identify_run, identify_init_run, args,
                              identify_args],
                     usage = "segway annotate-run [args] GENOMEDATA TRAINDIR ANNOTATEDIR")
    tasks.add_parser("annotate-finish", parents=[identify_finish, args,
                                                 identify_args],
                     usage = "segway annotate-finish [args] GENOMEDATA TRAINDIR ANNOTATEDIR")

    # posterior and identify take the same options
    tasks.add_parser("posterior-init", parents=[identify_init,
                                                identify_init_run, args,
                                                identify_args],
                     usage = "segway posterior-init [args] GENOMEDATA TRAINDIR POSTDIR")
    tasks.add_parser("posterior-run", parents=[identify_run, identify_init_run,
                                               args, identify_args],
                     usage = "segway posterior-run [args] GENOMEDATA TRAINDIR POSTDIR")
    tasks.add_parser("posterior-finish", parents=[identify_finish, args,
                                                  identify_args],
                     usage = "segway posterior-finish [args] GENOMEDATA TRAINDIR POSTDIR")

    tasks.add_parser("train",
                     parents=[train_init, train_run, train_init_run, train_finish, args],
                     usage="segway train [args] GENOMEDATA TRAINDIR")
    tasks.add_parser("annotate",
                     parents=[identify_init, identify_run, identify_init_run, identify_finish,
                              args, identify_args],
                     usage="segway annotate [args] GENOMEDATA TRAINDIR ANNOTATEDIR")
    tasks.add_parser("identify",
                     parents=[identify_init, identify_init_run, identify_run, identify_finish,
                              args, identify_args],
                     usage="segway identify [args] GENOMEDATA TRAINDIR ANNOTATEDIR")
    tasks.add_parser("posterior",
                     parents=[identify_init, identify_init_run, identify_run, identify_finish,
                              args, identify_args],
                     usage="segway posterior [args] GENOMEDATA TRAINDIR POSTDIR")
    tasks.add_parser("identify+posterior",
                     parents=[identify_init, identify_init_run, identify_run,
                              identify_finish, args, identify_args],
                     usage="segway identify+posterior [args] GENOMEDATA TRAINDIR POSTDIR")

    options = parser.parse_args(argv)

    # options is a non-iterable Namespace object
    # Separate arguments and task_spec from options
    task_spec = options.task_spec
    archives = options.archives
    # Add [0] to unlist directory names
    traindir = options.traindir[0]
    if hasattr(options, "annotatedir"):
        annotatedir = options.annotatedir[0]
    else:
        annotatedir = None

    return task_spec, options, archives, traindir, annotatedir


def check_gmtk_version():
    """ Checks if the supported minimum GMTK version is installed """
    # Typical expected output from "gmtkTriangulate -version":
    # gmtkTriangulate (GMTK) 1.4.3
    # Mercurial id: 8995e40101d2 tip
    # checkin date: Fri Oct 30 10:51:44 2015 -0700

    # Older versions:
    # GMTK 1.4.0 (Mercurial id: bdf2718cc6ce tip checkin date: Thu Jun 25 12:31:56 2015 -0700)

    # Try to open gmtkTriangulate to get the version
    try:
        output_string = TRIANGULATE_PROG.getoutput("-version").decode()
    # If GMTK was not found
    except IOError:  # optbuild raises an IOError when programs cannot be found
        # Raise a runtime error stating that GMTK was not found on the path
        raise RuntimeError("GMTK cannot be found on your PATH.\n"
                           "Please install GMTK from "
                           "https://github.com/melodi-lab/gmtk/releases "
                           "before running Segway.")

    output_lines = output_string.splitlines()

    # Check if there's only one line of output (for older versions)
    if len(output_lines) == 1:
        version_word_index = 1
    else:
        version_word_index = -1

    # Get the first line of output
    first_output_line = output_lines[0]

    # Get the version string from the proper word on the line
    current_gmtk_version_string = first_output_line.split()[version_word_index]

    # Get the version number to compare with the minimum version
    current_gmtk_version = parse_version(current_gmtk_version_string)

    if current_gmtk_version < MINIMUM_GMTK_VERSION:
        # Raise a runtime error stating the version found and the
        # minimum required
        raise RuntimeError(GMTK_VERSION_ERROR_MSG %
                           (current_gmtk_version_string,
                            MINIMUM_GMTK_VERSION))


def main(argv=sys.argv[1:]):

    task_spec, options, archives, traindir, annotatedir = parse_options(argv)

    check_gmtk_version()
    runner = Runner.fromoptions(task_spec, archives,
                                traindir, annotatedir, options)

    return runner()

if __name__ == "__main__":
    sys.exit(main())
