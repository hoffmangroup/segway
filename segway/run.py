#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: main Segway implementation
"""

__version__ = "$Revision$"

# Copyright 2008-2009 Michael M. Hoffman <mmh1@washington.edu>

from cStringIO import StringIO
from collections import defaultdict
from contextlib import closing, nested
from copy import copy
from datetime import datetime
from distutils.spawn import find_executable
from errno import EEXIST, ENOENT
from functools import partial
from heapq import heappop, heappush
from itertools import count, izip, repeat
from math import ceil, floor, frexp, ldexp, log, log10
from operator import neg
from os import environ, extsep
import re
from shutil import copy2
from string import Template
import sys
from threading import Event, Lock, Thread
from time import sleep
from uuid import uuid1

from drmaa import ExitTimeoutException, JobControlAction, JobState
from genomedata import Genome
from numpy import (amin, amax, append, arange, arcsinh, array, diagflat, empty,
                   finfo, float32, intc, NINF, ones, outer, sqrt, square, tile,
                   vectorize, vstack, where, zeros)
from numpy.random import uniform
from optplus import str2slice_or_int
from optbuild import AddableMixin
from path import path
from pkg_resources import Requirement, working_set
from tabdelim import DictReader, ListWriter

from .bed import read_native
from .cluster import Session, JobTemplateFactory, make_native_spec
from ._util import (ceildiv, data_filename, data_string,
                    DTYPE_OBS_INT, DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                    DISTRIBUTION_ASINH_NORMAL, EXT_GZ, GB, get_chrom_coords,
                    get_label_color, gzip_open, is_empty_array, ISLAND_BASE_NA,
                    ISLAND_LST_NA, load_coords, _make_continuous_cells, MB,
                    NamedTemporaryDir, OptionBuilder_GMTK, PKG,
                    _save_observations_chunk, VITERBI_PROG)

# set once per file run
UUID = uuid1().hex

# XXX: I should really get some sort of Enum for this, I think Peter
# Norvig has one
DISTRIBUTIONS = [DISTRIBUTION_NORM, DISTRIBUTION_GAMMA,
                 DISTRIBUTION_ASINH_NORMAL]
DISTRIBUTION_DEFAULT = DISTRIBUTION_ASINH_NORMAL
DISTRIBUTIONS_LIKE_NORM = frozenset([DISTRIBUTION_NORM,
                                     DISTRIBUTION_ASINH_NORMAL])

## XXX: should be options
MEAN_METHOD_UNIFORM = "uniform" # randomly pick from the range
MEAN_METHOD_ML_JITTER = "ml_jitter" # maximum likelihood, then jitter

# maximum likelihood, adjusted by no more than 0.2*sd
MEAN_METHOD_ML_JITTER_STD = "ml_jitter_std"
MEAN_METHODS = [MEAN_METHOD_UNIFORM, MEAN_METHOD_ML_JITTER,
                MEAN_METHOD_ML_JITTER_STD]
MEAN_METHOD = MEAN_METHOD_ML_JITTER_STD

COVAR_METHOD_MAX_RANGE = "max_range" # maximum range
COVAR_METHOD_ML_JITTER = "ml_jitter" # maximum likelihood, then jitter
COVAR_METHOD_ML = "ml" # maximum likelihood
COVAR_METHODS = [COVAR_METHOD_MAX_RANGE, COVAR_METHOD_ML_JITTER,
                 COVAR_METHOD_ML]
COVAR_METHOD = COVAR_METHOD_ML

MAX_WEIGHT_SCALE = 25
MIN_NUM_SEGS = 2
NUM_SEGS = MIN_NUM_SEGS
RULER_SCALE = 10
MAX_EM_ITERS = 100
TEMPDIR_PREFIX = PKG + "-"
COVAR_TIED = True # would need to expand to MC, MX to change
MAX_CHUNKS = 1000

TEMP_DIRPATH = path("/tmp")

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
LEN_SEG_EXPECTED = 100000
SWAP_ENDIAN = False
POSTERIOR_SCALE_FACTOR = 100.0

## option defaults
VERBOSITY = 0
PRIOR_STRENGTH = 0

FINFO_FLOAT32 = finfo(float32)
MACHEP_FLOAT32 = FINFO_FLOAT32.machep
TINY_FLOAT32 = FINFO_FLOAT32.tiny

SIZEOF_FLOAT32 = float32().nbytes
SIZEOF_DTYPE_OBS_INT = DTYPE_OBS_INT().nbytes
SIZEOF_INTC = intc().nbytes

JITTER_STD_BOUND = 0.2
FUDGE_EP = -17 # ldexp(1, -17) = ~1e-6
assert FUDGE_EP > MACHEP_FLOAT32

# binary
JITTER_ORDERS_MAGNITUDE = 5 # log10(2**5) = 1.5 decimal orders of magnitude

FUDGE_TINY = -ldexp(TINY_FLOAT32, 6)
ABSOLUTE_FUDGE = 0.001

LOG_LIKELIHOOD_DIFF_FRAC = 1e-5

NUM_SEQ_COLS = 2 # dinucleotide, presence_dinucleotide

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2

MAX_FRAMES = 2000000 # 2 million
MEM_USAGE_BUNDLE = 100*MB # XXX: should start using this again
MEM_USAGE_PROGRESSION = "2,3,4,6,8,10,12,14,15"

POSTERIOR_CLIQUE_INDICES = dict(p=1, c=1, e=1)

## defaults
RANDOM_STARTS = 1

# self->self, self->other
PROBS_FORCE_TRANSITION = array([0.0, 1.0])
PROBS_PREVENT_TRANSITION = array([1.0, 0.0])

# replace NAN with SENTINEL to avoid warnings
# XXX: replace with something negative and outlandish again
SENTINEL = float32(9.87654321)

CPP_DIRECTIVE_FMT = "-D%s=%s"

GMTK_INDEX_PLACEHOLDER = "@D"
NAME_BUNDLE_PLACEHOLDER = "bundle"

# programs
ENV_CMD = "/usr/bin/env"
BASH_CMD = "bash"

# XXX: need to differentiate this (prog) from prog.prog == progname throughout
TRIANGULATE_PROG = OptionBuilder_GMTK("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_GMTK("gmtkEMtrainNew")
POSTERIOR_PROG = OptionBuilder_GMTK("gmtkJT")

SPECIAL_TRACKNAMES = ["dinucleotide", "supervisionLabel"]

def extjoin(*args):
    return extsep.join(args)

# extensions and suffixes
EXT_BED = "bed"
EXT_BIN = "bin"
EXT_LIKELIHOOD = "ll"
EXT_LIST = "list"
EXT_FLOAT = "float32"
EXT_INT = "int"
EXT_LOG = "log"
EXT_OUT = "out"
EXT_PARAMS = "params"
EXT_POSTERIOR = "posterior"
EXT_SH = "sh"
EXT_TAB = "tab"
EXT_TXT = "txt"
EXT_TRIFILE = "trifile"
EXT_WIG = "wig"

def make_prefix_fmt(num):
    # make sure there are sufficient leading zeros
    return "%%0%dd." % (int(floor(log10(num))) + 1)

PREFIX_ACC = "acc"
PREFIX_CMDLINE_SHORT = "run"
PREFIX_CMDLINE_LONG = "details"
PREFIX_CMDLINE_TOP = "segway"
PREFIX_POSTERIOR = "posterior"

PREFIX_VITERBI = "viterbi"
PREFIX_LIKELIHOOD = "likelihood"
PREFIX_CHUNK = "chunk"
PREFIX_PARAMS = "params"
PREFIX_JT_INFO = "jt_info"
PREFIX_JOB_LOG = "jobs"

PREFIX_JOB_NAME_TRAIN = "emt"
PREFIX_JOB_NAME_VITERBI = "vit"
PREFIX_JOB_NAME_POSTERIOR = "jt"

SUFFIX_LIST = extsep + EXT_LIST
SUFFIX_OUT = extsep + EXT_OUT
SUFFIX_TRIFILE = extsep + EXT_TRIFILE

BED_FILEBASENAME = extjoin(PKG, EXT_BED, EXT_GZ) # "segway.bed.gz"
FLOAT_TABFILEBASENAME = extjoin("observations", EXT_TAB)

SUBDIRNAME_ACC = "accumulators"
SUBDIRNAME_AUX = "auxiliary"
SUBDIRNAME_LIKELIHOOD = "likelihood"
SUBDIRNAME_LOG = "log"
SUBDIRNAME_OBS = "observations"
SUBDIRNAME_PARAMS = "params"
SUBDIRNAME_POSTERIOR = "posterior"
SUBDIRNAME_VITERBI = "viterbi"

SUBDIRNAMES_EITHER = [SUBDIRNAME_AUX]
SUBDIRNAMES_TRAIN = [SUBDIRNAME_ACC, SUBDIRNAME_LIKELIHOOD,
                     SUBDIRNAME_PARAMS]
SUBDIRNAMES_IDENTIFY = [SUBDIRNAME_POSTERIOR, SUBDIRNAME_VITERBI]

FLOAT_TAB_FIELDNAMES = ["filename", "chunk_index", "chrom", "start", "end"]
JOB_LOG_FIELDNAMES = ["jobid", "jobname", "prog", "num_segs", "num_frames",
                      "maxvmem", "cpu", "exit_status"]

# templates and formats
RES_STR_TMPL = "segway.str.tmpl"
RES_INPUT_MASTER_TMPL = "input.master.tmpl"
RES_OUTPUT_MASTER = "output.master"
RES_DONT_TRAIN = "dont_train.list"
RES_INC_TMPL = "segway.inc.tmpl"
RES_SEG_TABLE = "seg_table.tab"
RES_WRAPPER = "segway-wrapper.sh"

DIRICHLET_FRAG = "dirichlet_segCountDown_seg_segTransition" \
    " 3 CARD_SEGCOUNTDOWN CARD_SEG CARD_SEGTRANSITION"

DENSE_CPT_START_SEG_FRAG = "start_seg 0 CARD_SEG"
DENSE_CPT_SEG_SEG_FRAG = "seg_seg 1 CARD_SEG CARD_SEG"
DIRICHLET_SEGCOUNTDOWN_SEG_SEGTRANSITION_FRAG = \
    "DirichletTable dirichlet_segCountDown_seg_segTransition"

DENSE_CPT_SEGCOUNTDOWN_SEG_SEGTRANSITION_FRAG = \
    "segCountDown_seg_segTransition" \
    " 2 CARD_SEGCOUNTDOWN CARD_SEG CARD_SEGTRANSITION"

DENSE_CPT_SEG_DINUCLEOTIDE_FRAG = \
    "seg_dinucleotide 1 CARD_SEG CARD_DINUCLEOTIDE"

MEAN_TMPL = "mean_${seg}_${track} 1 ${rand}"

COVAR_TMPL_TIED = "covar_${track} 1 ${rand}"
# XXX: unused
COVAR_TMPL_UNTIED = "covar_${seg}_${track} 1 ${rand}"

GAMMASCALE_TMPL = "gammascale_${seg}_${track} 1 1 ${rand}"
GAMMASHAPE_TMPL = "gammashape_${seg}_${track} 1 1 ${rand}"

MC_NORM_TMPL = "1 COMPONENT_TYPE_DIAG_GAUSSIAN" \
    " mc_${distribution}_${seg}_${track} mean_${seg}_${track} covar_${track}"
MC_GAMMA_TMPL = "1 COMPONENT_TYPE_GAMMA mc_gamma_${seg}_${track}" \
    " ${min_track} gammascale_${seg}_${track} gammashape_${seg}_${track}"
MC_TMPLS = {"norm": MC_NORM_TMPL,
            "gamma": MC_GAMMA_TMPL,
            "asinh_norm": MC_NORM_TMPL}

MX_TMPL = "1 mx_${seg}_${track} 1 dpmf_always" \
    " mc_${distribution}_${seg}_${track}"

NAME_COLLECTION_TMPL = "collection_seg_${track} ${num_segs}"
NAME_COLLECTION_CONTENTS_TMPL = "mx_${seg}_${track}"

TRACK_FMT = "browser position %s:%s-%s"
FIXEDSTEP_FMT = "fixedStep chrom=%s start=%s step=1 span=1"

WIG_ATTRS = dict(autoScale="off")
WIG_ATTRS_VITERBI = dict(name="%s.%s" % (PKG, UUID),
                         visibility="dense",
                         viewLimits="0:1",
                         itemRgb="on",
                         **WIG_ATTRS)
WIG_ATTRS_POSTERIOR = dict(type="wiggle_0",
                           viewLimits="0:100",
                           visibility="full",
                           yLineMark="50",
                           maxHeightPixels="101:101:11",
                           windowingFunction="mean",
                           **WIG_ATTRS)

WIG_NAME_POSTERIOR = "%s segment %%s" % PKG

WIG_DESC_VITERBI = "%s segmentation of %%s" % PKG
WIG_DESC_POSTERIOR = "%s posterior decoding segment %%s of %%%%s" % PKG

TRAIN_ATTRNAMES = ["input_master_filename", "params_filename",
                   "log_likelihood_filename"]
LEN_TRAIN_ATTRNAMES = len(TRAIN_ATTRNAMES)

COMMENT_POSTERIOR_TRIANGULATION = \
    "%% triangulation modified for posterior decoding by %s" % PKG

FIXEDSTEP_HEADER = "fixedStep chrom=%s start=%d step=%d span=%d"

CARD_SEGTRANSITION = 2

# training results
# XXX: this should really be a namedtuple, yuck
OFFSET_NUM_SEGS = 1
OFFSET_FILENAMES = 2 # where the filenames begin in the results
OFFSET_PARAMS_FILENAME = 3

SEG_TABLE_WIDTH = 3
OFFSET_START = 0
OFFSET_END = 1
OFFSET_STEP = 2

SUPERVISION_UNSUPERVISED = 0
SUPERVISION_SEMISUPERVISED = 1
SUPERVISION_SUPERVISED = 2

SUPERVISION_LABEL_OFFSET = 1

TERMINATE = JobControlAction.TERMINATE

FAILED = JobState.FAILED
DONE = JobState.DONE

# CLEAN_PERIOD in lsb.params, after which jobs are removed from
# mbatchd's memory default is 3600, multiplying by 0.5 for a margin of
# error
# XXX: check lsb.params for real value of CLEAN_PERIOD
CLEAN_SAFE_TIME = int(3600 * 0.5)

# 62 so that it's not in sync with the 10 second job wait sleep time
THREAD_START_SLEEP_TIME = 62 # XXX: this should become an option
MIN_JOB_WAIT_SLEEP_TIME = 5 # min time to wait between checking job status
MAX_JOB_WAIT_SLEEP_TIME = 10 # max time to wait between checking job status

RESOLUTION = 1

INDEX_BED_START = 1

## exceptions
class ChunkOverMemUsageLimit(Exception):
    pass

## functions
try:
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

def make_fixedstep_header(chrom, start, resolution):
    """
    this function expects 0-based coordinates
    it does the conversion to 1-based coordinates for you
    """
    start_1based = start+1

    # XXX: if there is an overhang of less than resolution, then
    # having step/span = resolution means the last datum in a chunk
    # will actually extend too far. There's no point in fixing this
    # now, since we want to switch to bedGraph eventually anyway
    return FIXEDSTEP_HEADER % (chrom, start_1based, resolution, resolution)

def make_wig_attr(key, value):
    if " " in value:
        value = '"%s"' % value

    return "%s=%s" % (key, value)

def make_wig_attrs(mapping):
    res = " ".join(make_wig_attr(key, value)
                   for key, value in mapping.iteritems())

    return "track %s" % res

def vstack_tile(array_like, reps):
    return tile(array_like, (reps, 1))

def extjoin_not_none(*args):
    return extjoin(*[str(arg) for arg in args
                     if arg is not None])

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

def convert_chunks(attrs, name):
    supercontig_start = attrs.start
    edges_array = getattr(attrs, name) + supercontig_start
    res = edges_array.tolist()

    # XXX: this is a hack that was necessary due to a bugs in
    # save_metadata.py <r295; you can remove it when all the data is
    # reloaded
    if isinstance(res, list):
        return res
    else:
        return [res]

def is_training_progressing(last_ll, curr_ll,
                            min_ll_diff_frac=LOG_LIKELIHOOD_DIFF_FRAC):
    # using x !< y instead of x >= y to give the right default answer
    # in the case of NaNs
    return not abs((curr_ll - last_ll)/last_ll) < min_ll_diff_frac

def resource_substitute(resourcename):
    return Template(data_string(resourcename)).substitute

def make_template_filename(filename, resource, dirname=None, clobber=False,
                           thread_index=None):
    """
    returns (filename, is_new)
    """
    if filename:
        if not clobber and path(filename).exists():
            return filename, False
        # else filename is unchanged
    else:
        resource_part = resource.rpartition(".tmpl")
        stem = resource_part[0] or resource_part[2]
        stem_part = stem.rpartition(extsep)
        prefix = stem_part[0]
        ext = stem_part[2]

        filebasename = extjoin_not_none(prefix, thread_index, ext)

        filename = path(dirname) / filebasename

    return filename, True

def save_template(filename, resource, mapping, dirname=None,
                  clobber=False, thread_index=None):
    """
    creates a temporary file if filename is None or empty
    """
    filename, is_new = make_template_filename(filename, resource, dirname,
                                              clobber, thread_index)

    if is_new:
        with open(filename, "w+") as outfile:
            tmpl = Template(data_string(resource))
            text = tmpl.substitute(mapping)

            outfile.write(text)

    return filename, is_new

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

def make_spec(name, items):
    header_lines = ["%s_IN_FILE inline" % name, str(len(items)), ""]

    indexed_items = ["%d %s" % indexed_item
                     for indexed_item in enumerate(items)]

    all_lines = header_lines + indexed_items

    return "\n".join(all_lines) + "\n"

def array2text(a):
    ndim = a.ndim
    if ndim == 1:
        return " ".join(map(str, a))
    else:
        delimiter = "\n" * (ndim-1)
        return delimiter.join(array2text(row) for row in a)

# XXX: consider making much of frag automatic
def make_table_spec(frag, table):
    return "\n".join([frag, array2text(table), ""])

# def make_dt_spec(num_tracks):
#     return make_spec("DT", ["%d seg_obs%d BINARY_DT" % (index, index)
#                             for index in xrange(num_tracks)])

def prob_transition_from_expected_len(length):
    # formula from Meta-MEME paper, Grundy WN et al. CABIOS 13:397
    # see also Reynolds SM et al. PLoS Comput Biol 4:e1000213
    # ("duration modeling")
    return length / (1 + length)

def make_name_collection_spec(num_segs, tracknames):
    substitute = Template(NAME_COLLECTION_TMPL).substitute
    substitute_contents = Template(NAME_COLLECTION_CONTENTS_TMPL).substitute

    items = []

    for track_index, track in enumerate(tracknames):
        mapping = dict(track=track, num_segs=num_segs)

        contents = [substitute(mapping)]
        for seg_index in xrange(num_segs):
            seg = "seg%d" % seg_index
            mapping = dict(seg=seg, track=track)

            contents.append(substitute_contents(mapping))
        items.append("\n".join(contents))

    return make_spec("NAME_COLLECTION", items)

re_posterior_entry = re.compile(r"^\d+: (\S+) seg\((\d+)\)=(\d+)$")
def parse_posterior(iterable):
    """
    a generator.
    yields tuples (index, label, prob)

    index: (int) frame index
    label: (int) segment label
    prob: (float) prob value
    """
    # ignores non-matching lines
    for line in iterable:
        m_posterior_entry = re_posterior_entry.match(line.rstrip())

        if m_posterior_entry:
            group = m_posterior_entry.group
            yield (int(group(2)), int(group(3)), float(group(1)))

def read_posterior(infile, num_frames, num_labels):
    """
    returns an array (num_frames, num_labels)
    """
    # XXX: should these be single precision?
    res = zeros((num_frames, num_labels))

    for frame_index, label, prob in parse_posterior(infile):
        if label >= num_labels:
            raise ValueError("saw label %s but num_labels is only %s"
                             % (label, num_labels))

        res[frame_index, label] = prob

    return res

def load_posterior_write_wig((chrom, start, end), resolution, num_labels,
                             infilename, outfiles):
    header = make_fixedstep_header(chrom, start, resolution)
    num_frames = ceildiv(end - start, resolution)

    for outfile in outfiles:
        print >>outfile, header

    with open(infilename) as infile:
        probs = read_posterior(infile, num_frames, num_labels)

    probs_rounded = empty(probs.shape, int)

    # scale, round, and cast to int
    (probs * POSTERIOR_SCALE_FACTOR).round(out = probs_rounded)

    # print array columns as text to each outfile
    for outfile, probs_rounded_label in zip(outfiles, probs_rounded.T):

        # can't use array.tofile() because outfile is a GzipFile
        for prob in probs_rounded_label:
            print >>outfile, prob

def set_cwd_job_tmpl(job_tmpl):
    job_tmpl.workingDirectory = path.getcwd()

def jitter_cell(cell):
    """
    adds some random noise
    """
    # get the binary exponent and subtract JITTER_ORDERS_MAGNITUDE
    # e.g. 3 * 2**10 --> 1 * 2**5
    max_noise = ldexp(1, frexp(cell)[1] - JITTER_ORDERS_MAGNITUDE)

    return cell + uniform(-max_noise, max_noise)

jitter = vectorize(jitter_cell)

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

    return orig_num_cliques

def make_mem_req(mem_usage):
    # double usage at this point
    mem_usage_gibibytes = ceil(mem_usage / GB)

    return "%dG" % mem_usage_gibibytes

def update_starts(starts, ends, new_starts, new_ends, thread_index):
    next_index = thread_index + 1

    starts[next_index:next_index] = new_starts
    ends[next_index:next_index] = new_ends

def add_observation(observations, resourcename, **kwargs):
    observations.append(resource_substitute(resourcename)(**kwargs))

class Mixin_Lockable(AddableMixin):
    def __init__(self, *args, **kwargs):
        self.lock = Lock()
        return AddableMixin.__init__(self, *args, **kwargs)

LockableDefaultDict = Mixin_Lockable + defaultdict

class RestartableJob(object):
    def __init__(self, session, job_tmpl_factory, global_mem_usage,
                 mem_usage_key):
        self.session = session
        self.job_tmpl_factory = job_tmpl_factory

        # last trial index tried
        self.trial_index = -1

        self.global_mem_usage = global_mem_usage
        self.mem_usage_key = mem_usage_key

        # (-num_segs, -num_frames) so minimum value will be largest
        # num_segs, num_frames
        self.sort_key = (-mem_usage_key[1], -mem_usage_key[2])

    def run(self):
        job_tmpl_factory = self.job_tmpl_factory

        global_mem_usage = self.global_mem_usage
        mem_usage_key = self.mem_usage_key
        trial_index = global_mem_usage[mem_usage_key]

        #print >>sys.stderr, ("self.trial_index=%d; trial_index=%d"
        #                     % (self.trial_index, trial_index))

        # if this index was tried before and unsuccessful, increment
        # and set global_mem_usage, controlling for race conditions
        if self.trial_index == trial_index:
            with global_mem_usage.lock:
                choices = [global_mem_usage[mem_usage_key], trial_index+1]
                trial_index = max(choices)

            global_mem_usage[mem_usage_key] = trial_index

        self.trial_index = trial_index

        job_template = job_tmpl_factory(trial_index)
        res = self.session.runJob(job_template)

        assert res

        res_req = job_tmpl_factory.res_req
        if not isinstance(res_req, basestring):
            res_req = " ".join(res_req)

        jobname = job_template.jobName

        print >>sys.stderr, "queued %s: %s (%s)" % (res, jobname, res_req)

        return res

class RestartableJobDict(dict):
    def __init__(self, session, job_log_file, *args, **kwargs):
        self.session = session
        self.job_log_file = job_log_file
        self.unqueued_jobs = []

        return dict.__init__(self, *args, **kwargs)

    def calc_sleep_time(self):
        clean_safe_sleep_time = CLEAN_SAFE_TIME / len(self)
        sleep_time = min(clean_safe_sleep_time, MAX_JOB_WAIT_SLEEP_TIME)

    def is_sleep_time_gt_min(self):
        return self.calc_sleep_time() > MIN_JOB_WAIT_SLEEP_TIME

    def _queue_unconditional(self, restartable_job):
        """queue unconditionally; don't do any checks"""
        jobid = restartable_job.run()
        self[jobid] = restartable_job

    def queue(self, restartable_job):
        # handle dry run case
        if restartable_job is None:
            return

        # protect against queuing more jobs than I want to poll
        if self.is_sleep_time_gt_min():
            self._queue_unconditional(restartable_job)
        else:
            sort_key = restartable_job.sort_key
            heappush(self.unqueued_jobs, (sort_key, restartable_job))

    def queue_unqueued_jobs(self):
        while self.unqueued_jobs and self.is_sleep_time_gt_min():
            self._queue_unconditional(heappop(self.unqueued_jobs)[1])

    def wait(self):
        session = self.session

        jobids = self.keys()

        while jobids:
            # check each job individually
            for jobid in jobids:
                sleep(self.calc_sleep_time())

                try:
                    job_info = session.wait(jobid, session.TIMEOUT_NO_WAIT)
                except ExitTimeoutException:
                    # job isn't done yet
                    continue

                if job_info.hasSignal:
                    exit_status = job_info.terminatedSignal
                    # just in case this edge case ever happens
                    if exit_status == 0:
                        exit_status = "hasSignal"
                elif job_info.wasAborted:
                    exit_status = "wasAborted"
                elif job_info.hasExited:
                    exit_status = job_info.exitStatus
                else:
                    # this happens when the exit status is unknown
                    exit_status = "noExit"

                # if the job queue is already full, this will probably
                # result in the job going to unqueued jobs for now
                if exit_status != 0:
                    self.queue(self[jobid])

                restartable_job = self[jobid]
                jobname = restartable_job.job_tmpl_factory.template.jobName

                prog, num_segs, num_frames = restartable_job.mem_usage_key

                resource_usage = job_info.resourceUsage
                try: # SGE
                    maxvmem = resource_usage["maxvmem"]
                except KeyError: # LSF
                    maxvmem = resource_usage["vmem"]

                cpu = resource_usage["cpu"]
                row = [jobid, jobname, prog, str(num_segs), str(num_frames),
                       maxvmem, cpu, str(exit_status)]
                print >>self.job_log_file, "\t".join(row)
                self.job_log_file.flush() # allow reading file now

                del self[jobid]

                self.queue_unqueued_jobs()

                # XXX: should be able to check
                # session.jobStatus(jobid) but this has problems
                # 1. it returns DONE even with non-zero exit status
                # is this an SGE or DRMAA bug? shouldn't a
                # non-zero exit status be a failure? or is that
                # just LSF?
                # 2. sometimes I can't get the jobStatus() of a completed job:
                # InvalidJobException: code 18: The job specified
                # by the 'jobid' does not exist. see versions prior to
                # SVN r425 for code

            jobids = self.keys()

class RandomStartThread(Thread):
    def __init__(self, runner, session, thread_index, num_segs):
        # keeps it from rewriting variables that will be used
        # later or in a different thread
        self.runner = copy(runner)

        self.session = session
        self.num_segs = num_segs
        self.thread_index = thread_index

        Thread.__init__(self)

    def run(self):
        self.runner.session = self.session
        self.runner.num_segs = self.num_segs
        self.runner.thread_index = self.thread_index
        self.result = self.runner.run_train_start()

def maybe_quote_arg(text):
    return '"%s"' % text.replace('"', r'\"')

def cmdline2text(cmdline=sys.argv):
    return " ".join(maybe_quote_arg(arg) for arg in cmdline)

def _log_cmdline(logfile, cmdline):
    print >>logfile, " ".join(map(quote_spaced_str, cmdline))

def make_weight_scale(scale):
    return "scale %f" % scale

re_num_cliques = re.compile(r"^Number of cliques = (\d+)$")
re_clique_info = re.compile(r"^Clique information: .*, (\d+) unsigned words ")
class Runner(object):
    def __init__(self, **kwargs):
        # filenames
        self.float_filelistpath = None
        self.int_filelistpath = None

        self.gmtk_include_filename = None
        self.gmtk_include_filename_relative = None
        self.input_master_filename = None
        self.structure_filename = None
        self.triangulation_filename = None
        self.posterior_triangulation_filename = None
        self.jt_info_filename = None
        self.job_log_filename = None
        self.seg_table_filename = None

        self.params_filename = None # actual params filename for this instance
        self.params_filenames = None # list of possible params filenames
        self.dirname = None
        self.is_dirname_temp = False
        self.log_likelihood_filename = None
        self.log_likelihood_log_filename = None
        self.dont_train_filename = None

        self.viterbi_filenames = None

        self.obs_dirname = None
        self.bed_filename = None

        self.include_coords_filename = None
        self.exclude_coords_filename = None

        self.posterior_clique_indices = POSTERIOR_CLIQUE_INDICES.copy()

        self.metadata_done = False

        self.triangulation_filename_is_new = None

        self.supervision_coords = None
        self.supervision_labels = None

        self.card_supervision_label = -1

        # default is 0
        self.global_mem_usage = LockableDefaultDict(int)

        # data
        # a "chunk" is what GMTK calls a segment
        self.num_chunks = None
        self.chunk_coords = None
        self.mins = None
        self.maxs = None
        self.tracknames = None
        self.num_free_params = None

        # variables
        self.num_segs = NUM_SEGS
        self.random_starts = RANDOM_STARTS
        self.len_seg_strength = PRIOR_STRENGTH
        self.distribution = DISTRIBUTION_DEFAULT
        self.max_em_iters = MAX_EM_ITERS
        self.max_frames = MAX_FRAMES

        # flags
        self.clobber = False
        self.triangulate = True
        self.train = True # EM train # this should become an int for num_starts
        self.posterior = True
        self.verbosity = VERBOSITY
        self.identify = True # viterbi
        self.dry_run = False
        self.use_dinucleotide = None

        # XXX: should be removed, it was for segway-res-usage
        self.skip_large_mem_usage = False

        # functions
        self.train_prog = None

        self.__dict__.update(kwargs)

    @classmethod
    def fromoptions(cls, args, options):
        res = cls()

        res.genomedata_dirname = args[0]
        res.dirname = options.directory
        res.obs_dirname = options.observations
        res.bed_filename = options.bed

        res.input_master_filename = options.input_master
        res.structure_filename = options.structure
        res.params_filenames = options.trainable_params
        res.dont_train_filename = options.dont_train
        res.include_coords_filename = options.include_coords
        res.exclude_coords_filename = options.exclude_coords
        res.seg_table_filename = options.seg_table

        res.supervision_filename = options.semisupervised
        if options.semisupervised:
            res.supervision_type = SUPERVISION_SEMISUPERVISED
        else:
            res.supervision_type = SUPERVISION_UNSUPERVISED

        res.distribution = options.distribution
        res.random_starts = options.random_starts
        res.len_seg_strength = options.prior_strength
        res.ruler_scale = options.ruler_scale
        res.resolution = options.resolution

        include_tracknames = options.track
        # will mess things up if it's there
        assert "supervisionLabel" not in include_tracknames
        res.include_tracknames = include_tracknames

        res.verbosity = options.verbosity
        # multiple lists to one
        res.user_native_spec = sum([opt.split(" ")
                                    for opt in options.cluster_opt], [])

        res.num_segs = options.num_labels

        mem_usage_list = map(float, options.mem_usage.split(","))

        # XXX: should do a ceil first
        res.mem_usage_progression = (array(mem_usage_list) * GB).astype(int)

        res.clobber = options.clobber
        res.train = not options.no_train
        res.identify = not options.no_identify
        res.posterior = not options.no_posterior
        res.dry_run = options.dry_run
        res.keep_going = options.keep_going
        res.max_frames = options.split_sequences

        return res

    def make_cpp_options(self, input_params_filename=None,
                         output_params_filename=None):
        directives = {}

        if input_params_filename:
            directives["INPUT_PARAMS_FILENAME"] = input_params_filename

        if output_params_filename:
            directives["OUTPUT_PARAMS_FILENAME"] = output_params_filename

        directives["CARD_SEG"] = self.num_segs
        directives["CARD_FRAMEINDEX"] = self.max_frames

        res = " ".join(CPP_DIRECTIVE_FMT % item
                       for item in directives.iteritems())

        if res:
            return res

        # default: return None

    def load_log_likelihood(self):
        with open(self.log_likelihood_filename) as infile:
            log_likelihood = float(infile.read().strip())

        info_criterion = self.calc_info_criterion(log_likelihood)

        with open(self.log_likelihood_log_filename, "a") as logfile:
            row = map(str, [log_likelihood, info_criterion])
            print >>logfile, "\t".join(row)

        return log_likelihood, info_criterion

    def load_include_exclude_coords(self):
        self.include_coords = load_coords(self.include_coords_filename)
        self.exclude_coords = load_coords(self.exclude_coords_filename)

    def load_seg_table(self):
        filename = self.seg_table_filename

        if filename is None:
            filename = data_filename("seg_table.tab")

        num_segs = self.num_segs
        if isinstance(num_segs, slice):
            # XXX: wait, what if the step isn't 1?
            num_segs = num_segs.end-1

        table = zeros((num_segs, SEG_TABLE_WIDTH), dtype=int)
        ruler_scale = self.ruler_scale
        table[:, OFFSET_STEP] = ruler_scale

        with open(filename) as infile:
            reader = DictReader(infile)

            # overwriting is allowed
            for row in reader:
                # XXX: factor out
                # get table_row_indexes
                label = row["label"]
                label_slice = str2slice_or_int(label)

                if isinstance(label_slice, slice) and label_slice.stop is None:
                    label_slice = slice(label_slice.start, num_segs,
                                        label_slice.step)

                table_row_indexes = slice2range(label_slice)

                # get slice
                len_slice = str2slice_or_int(row["len"])

                # XXX: eventually, should read ruler scale from file
                # instead of using as a command-line option
                assert len_slice.step == self.ruler_scale

                len_tuple = (len_slice.start, len_slice.stop, len_slice.step)
                len_row = zeros((SEG_TABLE_WIDTH))

                for item_index, item in enumerate(len_tuple):
                    if item is not None:
                        len_row[item_index] = item

                table[table_row_indexes] = len_row

        self.seg_table = table

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

        seg_countdowns_initial = seg_countdowns_both.max(axis=0)

        self.seg_countdowns_initial = seg_countdowns_initial
        self.card_seg_countdown = seg_countdowns_initial.max() + 1

    def generate_tmpl_mappings(self, segnames=None, tracknames=None):
        if segnames is None:
            segnames = ["seg%d" % seg_index
                        for seg_index in xrange(self.num_segs)]

        if tracknames is None:
            tracknames = self.tracknames

        num_tracks = len(tracknames)

        for seg_index, segname in enumerate(segnames):
            for track_index, trackname in enumerate(tracknames):
                yield dict(seg=segname, track=trackname,
                           seg_index=seg_index, track_index=track_index,
                           index=num_tracks*seg_index + track_index,
                           distribution=self.distribution)

    def make_filename(self, *exts, **kwargs):
        filebasename = extjoin_not_none(*exts)

        # add subdirname if it exists
        return self.dirpath / kwargs.get("subdirname", "") / filebasename

    def set_tracknames(self, chromosome):
        # XXXopt: a lot of stuff here repeated for every chromosome
        # unnecessarily
        tracknames = chromosome.tracknames_continuous

        # includes special tracks (like dinucleotide)
        include_tracknames_all = self.include_tracknames

        # XXX: some repetition in next two assignments
        tracknames_all = [trackname
                          for trackname in tracknames + SPECIAL_TRACKNAMES
                          if trackname in include_tracknames_all]

        include_tracknames = frozenset(trackname
                                       for trackname in include_tracknames_all
                                       if trackname not in SPECIAL_TRACKNAMES)

        if include_tracknames:
            indexed_tracknames = ((index, trackname)
                                  for index, trackname in enumerate(tracknames)
                                  if trackname in include_tracknames)

            # redefine tracknames:
            track_indexes, tracknames = zip(*indexed_tracknames)
            track_indexes = array(track_indexes)

            # check that there aren't any missing tracks
            # XXX: a more informative error message would be better
            # here, telling us the missing tracks
            if len(tracknames) != len(include_tracknames):
                missing_tracknames = include_tracknames.difference(tracknames)
                missing_tracknames_text = ", ".join(missing_tracknames)
                msg = "could not find tracknames: %s" % missing_tracknames_text
                raise ValueError(msg)

        elif include_tracknames_all:
            # there are special tracknames only
            tracknames = []
            track_indexes = array([], intc)
            # XXX: this is too late to keep the file from being opened
            # all this junk should be refactored earlier, it doesn't
            # need to run every contig
            self.float_filelistpath = None
        else:
            track_indexes = arange(len(tracknames))
            tracknames_all = tracknames

        # replace illegal characters in tracknames only, not tracknames_all
        tracknames = [trackname.replace(".", "_") for trackname in tracknames]

        if self.tracknames is None:
            self.tracknames = tracknames
            self.tracknames_all = tracknames_all
            self.track_indexes = track_indexes
            self.track_indexes_text = ",".join(map(str, track_indexes))
        elif (self.tracknames != tracknames
              or (self.track_indexes != track_indexes).any()):
            raise ValueError("all tracknames attributes must be identical")

        return track_indexes

    def set_jt_info_filename(self):
        if not self.jt_info_filename:
            self.jt_info_filename = \
                self.make_filename(PREFIX_JT_INFO, EXT_TXT,
                                   subdirname=SUBDIRNAME_LOG)

    def set_last_params_filename(self, params_filename):
        if params_filename is None:
            self.last_params_filename = None
        elif path(params_filename).exists():
            self.last_params_filename = params_filename
        else:
            # if it doesn't exist, then it's actually a new filename,
            # but the only time this will be used is when new is not
            # set. And this will only happen from the master thread.
            self.last_params_filename = None

    def set_params_filename(self, thread_index=None, new=False):
        """
        None means the final params file, not for any particular thread
        """
        # if this is not run and params_filename is
        # unspecified, then it won't be passed to gmtkViterbiNew

        # these are unexpected corner cases for now
        assert not ((thread_index is not None and not new)
                    or (thread_index is None and new))

        params_filenames = self.params_filenames
        num_params_filenames = len(params_filenames)

        if thread_index is None and num_params_filenames == 1:
            # special case if there is only one param filename set
            # otherwise generate "params.params" anew
            params_filename = params_filenames[0]
        elif thread_index is not None and num_params_filenames > thread_index:
            params_filename = params_filenames[thread_index]
        else:
            params_filename = None

        self.set_last_params_filename(params_filename)

        # make new filenames when new is set, or params_filename is
        # not set, or the file already exists and we are training
        if (new or not params_filename
            or (self.train and path(params_filename).exists())):
            params_filename = \
                self.make_filename(PREFIX_PARAMS, thread_index, EXT_PARAMS,
                                   subdirname=SUBDIRNAME_PARAMS)

        self.params_filename = params_filename

    def set_log_likelihood_filename(self, thread_index=None, new=False):
        if new or not self.log_likelihood_filename:
            log_likelihood_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, thread_index,
                                   EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.log_likelihood_filename = log_likelihood_filename

            self.log_likelihood_log_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, thread_index, EXT_TAB,
                                   subdirname=SUBDIRNAME_LOG)

    def make_triangulation_dirpath(self):
        res = self.dirpath / "triangulation"
        self.make_dir(res)

        self.triangulation_dirpath = res

    def make_output_dirpath(self, dirname, thread_index, clobber=None):
        res = self.dirpath / "output" / dirname / str(thread_index)
        self.make_dir(res, clobber)

        return res

    def set_output_dirpaths(self, thread_index, clobber=None):
        make_output_dirpath_custom = partial(self.make_output_dirpath,
                                             thread_index=thread_index,
                                             clobber=clobber)

        self.output_dirpath = make_output_dirpath_custom("o")
        self.error_dirpath = make_output_dirpath_custom("e")

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
        self.make_dir(self.dirpath / subdirname)

    def make_subdirs(self, subdirnames):
        for subdirname in subdirnames:
            self.make_subdir(subdirname)

    def make_obs_filelistpath(self, ext):
        return self.obs_dirpath / extjoin(ext, EXT_LIST)

    def make_obs_dir(self):
        obs_dirname = self.obs_dirname
        if obs_dirname:
            obs_dirpath = path(obs_dirname)
        else:
            obs_dirpath = self.dirpath / SUBDIRNAME_OBS
            self.obs_dirname = obs_dirpath

        self.obs_dirpath = obs_dirpath

        try:
            self.make_dir(obs_dirpath)
        except OSError, err:
            if not (err.errno == EEXIST and obs_dirpath.isdir()):
                raise

        self.float_filelistpath = self.make_obs_filelistpath(EXT_FLOAT)
        self.int_filelistpath = self.make_obs_filelistpath(EXT_INT)
        self.float_tabfilepath = obs_dirpath / FLOAT_TABFILEBASENAME

    def make_obs_filepath(self, dirpath, prefix, suffix):
        return dirpath / (prefix + suffix)

    def make_obs_filepaths(self, chrom, chunk_index, temp=False):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_CHUNKS))
        prefix = prefix_feature_tmpl % chunk_index

        if temp:
            dirpath = TEMP_DIRPATH
            prefix = "".join([prefix, UUID, extsep])
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

        if self.is_dirname_temp:
            return orig_filename
        else:
            orig_filepath = path(orig_filename)
            dirpath = self.dirpath / subdirname

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
                       card_presence=resolution+1,
                       card_segCountDown=self.card_seg_countdown,
                       card_supervisionLabel=self.card_supervision_label,
                       card_frameIndex=self.max_frames,
                       ruler_scale=ruler_scale_scaled)

        aux_dirpath = self.dirpath / SUBDIRNAME_AUX

        include_filename, self.gmtk_include_filename_is_new = \
            save_template(self.gmtk_include_filename, RES_INC_TMPL, mapping,
                          aux_dirpath, self.clobber)

        dirpath_trailing_slash = self.dirpath + "/"
        include_filename_relative = \
            include_filename.partition(dirpath_trailing_slash)[2]
        assert include_filename_relative

        self.gmtk_include_filename = include_filename
        self.gmtk_include_filename_relative = include_filename

        # XXX: disable until you figure out a good way of dealing with
        # includes from params/input.master as well

        #self.gmtk_include_filename_relative = include_filename_relative

    def make_weight_spec(self, multiplier):
        resolution = self.resolution
        if resolution == 1:
            return make_weight_scale(multiplier)
        else:
            return " | ".join(make_weight_scale(index * multiplier)
                              for index in xrange(resolution + 1))

    def make_conditionalparents_spec(self, track):
        spec = ('seg(0) using mixture collection("collection_seg_%s") '
                'mapping("map_parent")'
                % track)

        return " | ".join(["CONDITIONALPARENTS_NIL_CONTINUOUS"] +
                          [spec] * self.resolution)

    def save_structure(self):
        tracknames = self.tracknames
        num_tracks = self.num_tracks
        num_datapoints = self.num_datapoints

        if self.use_dinucleotide:
            max_num_datapoints_track = sum(self.chunk_lens)
        else:
            max_num_datapoints_track = num_datapoints.max()

        observation_items = []
        zipper = izip(count(), tracknames, num_datapoints)
        for track_index, track, num_datapoints_track in zipper:
            # relates current num_datapoints to total number of
            # possible positions. This is better than making the
            # highest num_datapoints equivalent to 1, because it
            # becomes easier to mix and match different tracks without
            # changing the weights of any of them

            # XXX: this should be done based on the minimum seg len in
            # the seg table instead
            # weight scale cannot be more than MAX_WEIGHT_SCALE to avoid
            # artifactual problems

            weight_multiplier = min(max_num_datapoints_track
                                    / num_datapoints_track, MAX_WEIGHT_SCALE)
            # weight_scale = 1.0
            # assert weight_scale == 1.0

            conditionalparents_spec = self.make_conditionalparents_spec(track)
            weight_spec = self.make_weight_spec(weight_multiplier)

            # XXX: should avoid a weight line at all when weight_scale == 1.0
            # might avoid some extra multiplication in GMTK
            add_observation(observation_items, "observation.tmpl",
                            track=track, track_index=track_index,
                            presence_index=num_tracks+track_index,
                            conditionalparents_spec=conditionalparents_spec,
                            weight_spec=weight_spec)

        next_int_track_index = num_tracks*2
        # XXX: duplicative
        if self.use_dinucleotide:
            add_observation(observation_items, "dinucleotide.tmpl",
                            track_index=next_int_track_index,
                            presence_index=next_int_track_index+1)
            next_int_track_index += 2

        if self.supervision_type != SUPERVISION_UNSUPERVISED:
            add_observation(observation_items, "supervision.tmpl",
                            track_index=next_int_track_index)
            next_int_track_index += 1

        assert observation_items # must be at least one track
        observations = "\n".join(observation_items)

        mapping = dict(include_filename=self.gmtk_include_filename_relative,
                       observations=observations)

        self.structure_filename, self.structure_filename_is_new = \
            save_template(self.structure_filename, RES_STR_TMPL, mapping,
                          self.dirname, self.clobber)

    def save_observations_chunk(self, float_filename, int_filename, float_data,
                                seq_data=None, supervision_data=None):
        return _save_observations_chunk(float_filename, int_filename,
                                        float_data, self.resolution,
                                        self.distribution, seq_data,
                                        supervision_data)

    def subset_metadata_attr(self, name):
        subset_array = getattr(self, name)[self.track_indexes]

        setattr(self, name, subset_array)

    def subset_metadata(self):
        """
        limits all the metadata attributes to only tracks that are used
        """
        if self.metadata_done:
            return

        subset_metadata_attr = self.subset_metadata_attr
        subset_metadata_attr("mins")
        subset_metadata_attr("maxs")
        subset_metadata_attr("sums")
        subset_metadata_attr("sums_squares")
        subset_metadata_attr("num_datapoints")

        self.metadata_done = True # avoid repetition

    def accum_metadata(self, mins, maxs, sums, sums_squares, num_datapoints):
        if self.mins is None:
            self.mins = mins
            self.maxs = maxs
            self.sums = sums
            self.sums_squares = sums_squares
            self.num_datapoints = num_datapoints
        else:
            self.mins = amin([mins, self.mins], 0)
            self.maxs = amax([maxs, self.maxs], 0)
            self.sums += sums
            self.sums_squares += sums_squares
            self.num_datapoints += num_datapoints

    def accum_metadata_chromosome(self, chromosome):
        """
        accumulate metadata for a chromsome to the Runner instance

        returns True if there is metadata
        returns False if there is not
        """

        try:
            mins = chromosome.mins
            maxs = chromosome.maxs
            sums = chromosome.sums
            sums_squares = chromosome.sums_squares
            num_datapoints = chromosome.num_datapoints
        except AttributeError:
            # this means there is no data for that chromosome
            return False

        if not self.metadata_done:
            self.accum_metadata(mins, maxs, sums, sums_squares, num_datapoints)

        return True

    def get_progs_used(self):
        res = []

        if self.train:
            res.append(EM_TRAIN_PROG)
        if self.identify:
            res.append(VITERBI_PROG)
        if self.posterior:
            res.append(POSTERIOR_PROG)

        return res

    def write_observations(self, float_filelist, int_filelist, float_tabfile):
        # XXX: these expect different filepaths
        assert not ((self.train or self.posterior) and self.identify)

        print_obs_filepaths_custom = partial(self.print_obs_filepaths,
                                             float_filelist, int_filelist,
                                             temp=self.identify)
        save_observations_chunk = self.save_observations_chunk

        float_tabwriter = ListWriter(float_tabfile)
        float_tabwriter.writerow(FLOAT_TAB_FIELDNAMES)

        zipper = izip(count(), self.used_supercontigs, self.chunk_coords)

        for chunk_index, supercontig, (chrom, start, end) in zipper:
            float_filepath, int_filepath = \
                print_obs_filepaths_custom(chrom, chunk_index)

            row = [float_filepath, str(chunk_index), chrom, str(start),
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

                if not (self.train or self.posterior):
                    # don't actually write data
                    continue

                continuous_cells = \
                    self.make_continuous_cells(supercontig, start, end)

                save_observations_chunk(float_filepath, int_filepath,
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

        # chunk_start: relative to the beginning of the
        # supercontig
        chunk_start = start - supercontig.start
        chunk_end = end - supercontig.start

        if chunk_end < len_seq:
            return seq[chunk_start:chunk_end+1]
        elif chunk_end == len(seq):
            seq_chunk = seq[chunk_start:chunk_end]
            return append(seq_chunk, ord("N"))
        else:
            raise ValueError("sequence too short for supercontig")

    def prep_observations(self):
        # XXX: this function is way too long, try to get it to fit
        # inside a screen on your enormous monitor

        # this function is repeatable after max_mem_usage is discovered
        include_coords = self.include_coords
        exclude_coords = self.exclude_coords

        num_tracks = None # this is before any subsetting
        chunk_index = 0
        chunk_coords = []
        num_bases = 0
        used_supercontigs = [] # continuous objects
        max_frames = self.max_frames

        use_dinucleotide = "dinucleotide" in self.include_tracknames
        self.use_dinucleotide = use_dinucleotide

        # XXX: move this outside this function so self.genome can be
        # reused, and have a with statement

        # XXX: use groupby(include_coords) and then access chromosomes
        # randomly rather than iterating through them all
        self.genome = Genome(self.genomedata_dirname)
        for chromosome in self.genome:
            chrom = chromosome.name

            chr_include_coords = get_chrom_coords(include_coords, chrom)
            chr_exclude_coords = get_chrom_coords(exclude_coords, chrom)

            if (chr_include_coords is not None
                and is_empty_array(chr_include_coords)):
                continue

            # if there is no metadata, then skip the chromosome
            if not self.accum_metadata_chromosome(chromosome):
                continue

            track_indexes = self.set_tracknames(chromosome)
            num_tracks = len(track_indexes)

            self.set_num_tracks(num_tracks)

            if num_tracks:
                supercontigs = chromosome.itercontinuous()
            else:
                supercontigs = izip(chromosome, repeat(None))

            for supercontig, continuous in supercontigs:
                assert continuous is None or continuous.shape[1] >= num_tracks

                if continuous is None:
                    starts = [supercontig.start]
                    ends = [supercontig.end]
                else:
                    attrs = supercontig.attrs
                    starts = convert_chunks(attrs, "chunk_starts")
                    ends = convert_chunks(attrs, "chunk_ends")

                ## iterate through chunks and write
                ## izip so it can be modified in place
                for thread_index, start, end in izip(count(), starts, ends):
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
                                          thread_index)
                            continue # consider the newly split sequences next

                    num_bases_chunk = end - start
                    num_frames = ceildiv(num_bases_chunk, self.resolution)
                    if not MIN_FRAMES <= num_frames:
                        text = " skipping short segment of length %d" % num_frames
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
                                      thread_index)
                        continue

                    # start: relative to beginning of chromosome
                    chunk_coords.append((chrom, start, end))
                    used_supercontigs.append(supercontig)

                    num_bases += num_bases_chunk

                    chunk_index += 1

        self.subset_metadata()

        self.num_chunks = chunk_index # already has +1 added to it
        self.num_bases = num_bases
        self.chunk_coords = chunk_coords

        self.used_supercontigs = used_supercontigs

    def set_num_tracks(self, num_tracks):
        num_int_cols = num_tracks

        if self.use_dinucleotide:
            num_int_cols += NUM_SEQ_COLS
        if self.supervision_type != SUPERVISION_UNSUPERVISED:
            num_int_cols += 1

        self.num_tracks = num_tracks
        self.num_int_cols = num_int_cols

    def open_writable_or_dummy(self, filepath, clobber=None):
        if clobber is None:
            clobber = self.clobber

        if not filepath or (not clobber and filepath.exists()):
            return closing(StringIO()) # dummy output
        else:
            return open(filepath, "w")

    def save_observations(self, clobber=None):
        open_writable = partial(self.open_writable_or_dummy, clobber=clobber)

        with open_writable(self.float_filelistpath) as float_filelist:
            with open_writable(self.int_filelistpath) as int_filelist:
                with open_writable(self.float_tabfilepath) as float_tabfile:
                    self.write_observations(float_filelist, int_filelist,
                                            float_tabfile)

    def calc_means_vars(self):
        num_datapoints = self.num_datapoints
        means = self.sums / num_datapoints
        distribution = self.distribution

        if distribution == DISTRIBUTION_ASINH_NORMAL:
            means = arcsinh(means)

        # this is an unstable way of calculating the variance,
        # but it should be good enough
        # Numerical Recipes in C, Eqn 14.1.7
        # XXX: best would be to switch to the pairwise parallel method
        # (see Wikipedia)
        sums_squares_normalized = self.sums_squares / num_datapoints
        if distribution == DISTRIBUTION_ASINH_NORMAL:
            sums_squares_normalized = arcsinh(sums_squares_normalized)

        self.means = means
        self.vars = sums_squares_normalized - square(means)

    def get_track_lt_min(self, track_index):
        """
        returns a value less than a minimum in a track
        """
        # XXX: refactor into a new function
        min_track = self.mins[track_index]

        # fudge the minimum by a very small amount. this is not
        # continuous, but hopefully we won't get values where it
        # matters
        # XXX: restore this after GMTK issues fixed
        # if min_track == 0.0:
        #     min_track_fudged = FUDGE_TINY
        # else:
        #     min_track_fudged = min_track - ldexp(abs(min_track), FUDGE_EP)

        # this happens for really big numbers or really small
        # numbers; you only have 7 orders of magnitude to play
        # with on a float32
        min_track_f32 = float32(min_track)

        assert min_track_f32 - float32(ABSOLUTE_FUDGE) != min_track_f32
        return min_track - ABSOLUTE_FUDGE

    def make_items_multiseg(self, tmpl, data=None, segnames=None):
        substitute = Template(tmpl).substitute

        res = []
        for mapping in self.generate_tmpl_mappings(segnames):
            track_index = mapping["track_index"]

            if self.distribution == DISTRIBUTION_GAMMA:
                mapping["min_track"] = self.get_track_lt_min(track_index)

            if data is not None:
                seg_index = mapping["seg_index"]
                mapping["rand"] = data[seg_index, track_index]

            res.append(substitute(mapping))

        return res

    def make_spec_multiseg(self, name, *args, **kwargs):
        return make_spec(name, self.make_items_multiseg(*args, **kwargs))

    def make_empty_cpt(self):
        num_segs = self.num_segs

        return zeros((num_segs, num_segs))

    def make_linked_cpt(self):
        """
        has all of the seg0_0 -> seg0_1 links, etc.

        also has a seg 0_f -> seg1_0 link which must be overwritten,
        such as by add_cpt()
        """
        return diagflat(ones(self.num_segs-1), 1)

    def add_final_probs_to_cpt(self, cpt):
        """
        modifies original input
        """
        num_segs = self.num_segs

        prob_self_self = 0.0
        prob_self_other = 1.0 / (num_segs - 1)

        # set everywhere (including diagonals to be rewritten)
        cpt[...] = prob_self_other

        # set diagonal
        range_cpt = xrange(num_segs)
        cpt[range_cpt, range_cpt] = prob_self_self

        return cpt

    def make_dirichlet_table(self):
        num_segs = self.num_segs

        probs = self.make_dense_cpt_segCountDown_seg_segTransition()

        # XXX: the ratio is not exact as num_bases is not the same as
        # the number of base-base transitions. It is surely close
        # enough, though
        total_pseudocounts = self.len_seg_strength * self.num_bases
        divisor = self.card_seg_countdown * self.num_segs
        pseudocounts_per_row = total_pseudocounts / divisor

        # astype(int) means flooring the floats
        pseudocounts = (probs * pseudocounts_per_row).astype(int)

        return pseudocounts

    def make_dirichlet_spec(self):
        dirichlet_table = self.make_dirichlet_table()
        items = [make_table_spec(DIRICHLET_FRAG, dirichlet_table)]

        return make_spec("DIRICHLET_TAB", items)

    def make_dense_cpt_start_seg_spec(self):
        cpt = zeros((1, self.num_segs))

        cpt[0, :] = 1.0 / self.num_segs

        return make_table_spec(DENSE_CPT_START_SEG_FRAG, cpt)

    def make_dense_cpt_seg_seg_spec(self):
        num_segs = self.num_segs

        cpt = self.add_final_probs_to_cpt(self.make_linked_cpt())
        return make_table_spec(DENSE_CPT_SEG_SEG_FRAG, cpt)

    def make_dinucleotide_table_row(self):
        # simple one-parameter model
        gc = uniform()
        at = 1 - gc

        a = at / 2
        c = gc / 2
        g = gc - c
        t = 1 - a - c - g

        acgt = array([a, c, g, t])

        # shape: (16,)
        return outer(acgt, acgt).ravel()

    def make_dense_cpt_seg_dinucleotide_spec(self):
        table = [self.make_dinucleotide_table_row()
                 for seg_index in xrange(self.num_segs)]

        return make_table_spec(DENSE_CPT_SEG_DINUCLEOTIDE_FRAG, table)

    def make_dense_cpt_segCountDown_seg_segTransition(self):
        card_seg_countdown = self.card_seg_countdown

        # by default, when segCountDown is high, never transition
        res = empty((card_seg_countdown, self.num_segs, CARD_SEGTRANSITION))

        # set probs_allow_transition
        len_seg_expected_scaled = LEN_SEG_EXPECTED // self.resolution
        prob_self_self = \
            prob_transition_from_expected_len(len_seg_expected_scaled)
        prob_self_other = 1.0 - prob_self_self
        probs_allow_transition = array([prob_self_self, prob_self_other])

        # find the labels with maximum segment lengths and those without
        table = self.seg_table
        ends = table[:, OFFSET_END]
        bitmap_without_maximum = ends == 0

        # where() returns a tuple; this unpacks it
        labels_with_maximum, = where(~bitmap_without_maximum)
        labels_without_maximum, = where(bitmap_without_maximum)

        # labels without a maximum
        res[0, labels_without_maximum] = probs_allow_transition
        res[1:, labels_without_maximum] = PROBS_PREVENT_TRANSITION

        # labels with a maximum
        seg_countdowns_initial = self.seg_countdowns_initial

        res[0, labels_with_maximum] = PROBS_FORCE_TRANSITION
        for label in labels_with_maximum:
            seg_countdown_initial = seg_countdowns_initial[label]
            minimum = table[label, OFFSET_START] // table[label, OFFSET_STEP]

            seg_countdown_allow = seg_countdown_initial - minimum + 1

            res[1:seg_countdown_allow, label] = probs_allow_transition
            res[seg_countdown_allow:, label] = PROBS_PREVENT_TRANSITION

        return res

    def make_dense_cpt_segCountDown_seg_segTransition_spec(self):
        cpt = self.make_dense_cpt_segCountDown_seg_segTransition()

        if self.len_seg_strength > 0:
            frag = "\n".join([DENSE_CPT_SEGCOUNTDOWN_SEG_SEGTRANSITION_FRAG,
                              DIRICHLET_SEGCOUNTDOWN_SEG_SEGTRANSITION_FRAG])
        else:
            frag = DENSE_CPT_SEGCOUNTDOWN_SEG_SEGTRANSITION_FRAG

        return make_table_spec(frag, cpt)

    def make_dense_cpt_spec(self):
        num_segs = self.num_segs

        items = [self.make_dense_cpt_start_seg_spec(),
                 self.make_dense_cpt_seg_seg_spec(),
                 self.make_dense_cpt_segCountDown_seg_segTransition_spec()]

        if self.use_dinucleotide:
            items.append(self.make_dense_cpt_seg_dinucleotide_spec())

        return make_spec("DENSE_CPT", items)

    def rand_means(self):
        low = self.mins
        high = self.maxs
        num_segs = self.num_segs

        assert len(low) == len(high)

        # size parameter is so that we always get an array, even if it
        # has shape = (1,)

        # iterator so that we regenerate uniform for every seg
        return array([uniform(low, high, len(low))
                      for seg_index in xrange(num_segs)])

    def make_means(self):
        num_segs = self.num_segs
        means = self.means

        if MEAN_METHOD == MEAN_METHOD_UNIFORM:
            return self.rand_means()
        elif MEAN_METHOD == MEAN_METHOD_ML_JITTER:
            return jitter(vstack_tile(means, num_segs))
        elif MEAN_METHOD == MEAN_METHOD_ML_JITTER_STD:
            stds = sqrt(self.vars)

            means_tiled = vstack_tile(means, num_segs)
            stds_tiled = vstack_tile(stds, num_segs)

            noise = uniform(-JITTER_STD_BOUND, JITTER_STD_BOUND,
                             stds_tiled.shape)

            return means_tiled + (stds_tiled * noise)

        raise ValueError("unsupported MEAN_METHOD")

    def make_mean_spec(self):
        return self.make_spec_multiseg("MEAN", MEAN_TMPL, self.make_means())

    def make_covars(self):
        num_segs = self.num_segs

        if COVAR_METHOD == COVAR_METHOD_MAX_RANGE:
            ranges = self.maxs - self.mins
            return vstack_tile(ranges, num_segs)
        elif COVAR_METHOD == COVAR_METHOD_ML_JITTER:
            return jitter(vstack_tile(self.vars, num_segs))
        elif COVAR_METHOD == COVAR_METHOD_ML:
            return vstack_tile(self.vars, num_segs)

        raise ValueError("unsupported COVAR_METHOD")

    def make_covar_spec(self, tied):
        if tied:
            segnames = ["any"]
            tmpl = COVAR_TMPL_TIED
        else:
            segnames = None
            tmpl = COVAR_TMPL_UNTIED

        vars = self.make_covars()

        return self.make_spec_multiseg("COVAR", tmpl, vars, segnames)

    def make_items_gamma(self):
        means = self.means
        vars = self.vars

        substitute_scale = Template(GAMMASCALE_TMPL).substitute
        substitute_shape = Template(GAMMASHAPE_TMPL).substitute

        # random start values are equivalent to the random start
        # values of a Gaussian:
        #
        # means = scales * shapes
        # vars = shapes * scales**2
        #
        # therefore:
        scales = vars / means
        shapes = means**2 / vars

        res = []
        for mapping in self.generate_tmpl_mappings():
            seg_index = mapping["seg_index"]
            track_index = mapping["track_index"]
            index = mapping["index"] * 2

            # factory for new dictionaries that start with mapping
            mapping_plus = partial(dict, **mapping)

            scale = jitter(scales[track_index])
            shape = jitter(shapes[track_index])

            mapping_scale = mapping_plus(rand=scale, index=index)
            res.append(substitute_scale(mapping_scale))

            mapping_shape = mapping_plus(rand=shape, index=index+1)
            res.append(substitute_shape(mapping_shape))

        return res

    def make_gamma_spec(self):
        return make_spec("REAL_MAT", self.make_items_gamma())

    def make_mc_spec(self):
        return self.make_spec_multiseg("MC", MC_TMPLS[self.distribution])

    def make_mx_spec(self):
        return self.make_spec_multiseg("MX", MX_TMPL)

    def make_segCountDown_tree_spec(self, resourcename):
        num_segs = self.num_segs
        table = self.seg_table
        seg_countdowns_initial = self.seg_countdowns_initial

        header = ([str(num_segs)] +
                  [str(num_seg) for num_seg in xrange(num_segs-1)] +
                  ["default"])

        lines = [" ".join(header)]

        for seg, seg_countdown_initial in enumerate(seg_countdowns_initial):
            lines.append("    -1 %d" % seg_countdown_initial)

        tree = "\n".join(lines)

        return resource_substitute(resourcename)(tree=tree)

    def make_map_seg_segCountDown_dt_spec(self):
        return self.make_segCountDown_tree_spec("map_seg_segCountDown.dt.tmpl")

    def make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec(self):
        return self.make_segCountDown_tree_spec("map_segTransition_ruler_seg_segCountDown_segCountDown.dt.tmpl")

    def make_items_dt(self):
        res = []

        res.append(data_string("map_parent.dt.txt"))
        res.append(data_string("map_frameIndex_ruler.dt.txt"))
        res.append(self.make_map_seg_segCountDown_dt_spec())
        res.append(self.make_map_segTransition_ruler_seg_segCountDown_segCountDown_dt_spec())

        supervision_type = self.supervision_type
        if supervision_type == SUPERVISION_SEMISUPERVISED:
            res.append(data_string("map_supervisionLabel_seg_alwaysTrue_semisupervised.dt.txt"))
        elif supervision_type == SUPERVISION_SUPERVISED:
             # XXX: does not exist yet
            res.append(data_string("map_supervisionLabel_seg_alwaysTrue_supervised.dt.txt"))
        else:
            assert supervision_type == SUPERVISION_UNSUPERVISED

        return res

    def make_dt_spec(self):
        return make_spec("DT", self.make_items_dt())

    def save_input_master(self, thread_index=None, new=False):
        num_free_params = 0

        tracknames = self.tracknames
        num_segs = self.num_segs
        num_tracks = self.num_tracks

        include_filename = self.gmtk_include_filename_relative

        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        dt_spec = self.make_dt_spec()

        if self.len_seg_strength > 0:
            dirichlet_spec = self.make_dirichlet_spec()
        else:
            dirichlet_spec = ""

        dense_cpt_spec = self.make_dense_cpt_spec()

        # seg_seg
        num_free_params += num_segs * (num_segs - 1)

        # segCountDown_seg_segTransition
        num_free_params += num_segs

        self.calc_means_vars()

        distribution = self.distribution
        if distribution in DISTRIBUTIONS_LIKE_NORM:
            mean_spec = self.make_mean_spec()
            covar_spec = self.make_covar_spec(COVAR_TIED)
            gamma_spec = ""

            if COVAR_TIED:
                num_free_params += (num_segs + 1) * num_tracks
            else:
                num_free_params += (num_segs * 2) * num_tracks
        elif distribution == DISTRIBUTION_GAMMA:
            mean_spec = ""
            covar_spec = ""

            # XXX: another option is to calculate an ML estimate for
            # the gamma distribution rather than the ML estimate for the
            # mean and converting
            gamma_spec = self.make_gamma_spec()

            num_free_params += (num_segs * 2) * num_tracks
        else:
            raise ValueError("distribution %s not supported" % distribution)

        mc_spec = self.make_mc_spec()
        mx_spec = self.make_mx_spec()
        name_collection_spec = make_name_collection_spec(num_segs, tracknames)
        card_seg = num_segs

        params_dirpath = self.dirpath / SUBDIRNAME_PARAMS

        self.input_master_filename, input_master_filename_is_new = \
            save_template(input_master_filename, RES_INPUT_MASTER_TMPL,
                          locals(), params_dirpath, self.clobber,
                          thread_index)

        # print >>sys.stderr, "input_master_filename = %s; is_new = %s" \
        #     % (self.input_master_filename, input_master_filename_is_new)

        # only use num_free_params if a new input.master was created
        if input_master_filename_is_new:
            # print >>sys.stderr, "num_free_params = %d" % num_free_params
            self.num_free_params = num_free_params

    def save_dont_train(self):
        self.dont_train_filename = self.save_resource(RES_DONT_TRAIN,
                                                      SUBDIRNAME_AUX)

    def save_output_master(self):
        self.output_master_filename = self.save_resource(RES_OUTPUT_MASTER,
                                                         SUBDIRNAME_PARAMS)

    def make_viterbi_filenames(self):
        dirpath = self.dirpath / SUBDIRNAME_VITERBI
        num_chunks = self.num_chunks

        viterbi_filename_fmt = (PREFIX_VITERBI + make_prefix_fmt(num_chunks)
                                + EXT_BED)
        viterbi_filenames = [dirpath / viterbi_filename_fmt % index
                             for index in xrange(num_chunks)]

        self.viterbi_filenames = viterbi_filenames

    def make_posterior_filenames(self):
        make_posterior_filename = self.make_posterior_filename
        chunk_range = xrange(self.num_chunks)

        self.posterior_filenames = map(make_posterior_filename, chunk_range)

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
                supervision_coords_chrom = supervision_coords[chrom]
                start = datum.chromStart
                end = datum.chromEnd

                for coord_start, coord_end in supervision_coords_chrom:
                    # disallow overlaps
                    assert coord_start >= end or coord_end <= start

                supervision_coords_chrom.append((start, end))
                supervision_labels[chrom].append(int(datum.name))

        max_supervision_label = max(max(labels)
                                    for labels
                                    in supervision_labels.itervalues())

        self.supervision_coords = supervision_coords
        self.supervision_labels = supervision_labels

        self.include_tracknames.append("supervisionLabel")
        self.card_supervision_label = (max_supervision_label + 1 +
                                       SUPERVISION_LABEL_OFFSET)

    def save_params(self):
        self.load_include_exclude_coords()

        self.make_subdirs(SUBDIRNAMES_EITHER)
        self.make_obs_dir()

        self.load_supervision()

        # do first, because it sets self.num_tracks and self.tracknames
        self.prep_observations()

        # sets self.chunk_lens, needed for save_structure() to do
        # Dirichlet stuff (but rewriting structure is unnecessary)
        self.make_chunk_lens()

        self.load_seg_table()

        self.save_include()
        self.save_structure()
        self.set_params_filename()

        train = self.train
        identify = self.identify
        posterior = self.posterior

        if train or identify or posterior:
            self.set_jt_info_filename()

        if train:
            self.make_subdirs(SUBDIRNAMES_TRAIN)

            if not self.dont_train_filename:
                self.save_dont_train()

            self.save_output_master()

            # might turn off self.train, if the params already exist
            self.set_log_likelihood_filename()

        self.save_observations()

        if identify or posterior:
            self.make_subdirs(SUBDIRNAMES_IDENTIFY)

            # this requires the number of observations
            self.make_viterbi_filenames()

    def resave_params(self):
        """
        repeats necessary parts after an adjustment to self.mem_per_obs
        """
        self.prep_observations()
        self.make_chunk_lens()
        self.save_observations(clobber=True)

        if self.identify or self.posterior:
            # overwrite previous
            self.make_viterbi_filenames()

    def copy_results(self, name, src_filename, dst_filename):
        if dst_filename:
            copy2(src_filename, dst_filename)
        else:
            dst_filename = src_filename

        setattr(self, name, dst_filename)

    def make_posterior_wig_filename(self, seg):
        seg_label = ("seg%s" % make_prefix_fmt(self.num_segs)[:-1]) % seg

        return self.make_filename(PREFIX_POSTERIOR, seg_label, EXT_WIG, EXT_GZ)

    def make_wig_desc_attrs(self, mapping, desc_tmpl):
        attrs = mapping.copy()
        attrs["description"] = desc_tmpl % ", ".join(self.tracknames_all)

        return make_wig_attrs(attrs)

    def make_wig_header_viterbi(self):
        return self.make_wig_desc_attrs(WIG_ATTRS_VITERBI, WIG_DESC_VITERBI)

    def make_wig_header_posterior(self, seg_label):
        attrs = WIG_ATTRS_POSTERIOR.copy()
        attrs["name"] = WIG_NAME_POSTERIOR % seg_label
        attrs["color"] = get_label_color(seg_label)

        return self.make_wig_desc_attrs(attrs,
                                        WIG_DESC_POSTERIOR % seg_label)

    def concatenate_bed(self):
        # the final bed filename, not the individual viterbi_filenames
        bed_filename = self.bed_filename

        if bed_filename is None:
            bed_filename = self.dirpath / BED_FILEBASENAME

        # values for comparison to combine adjoining segments
        last_line = ""
        last_start = None
        last_vals = (None, None, None) # (chrom, coord, seg)

        with gzip_open(bed_filename, "w") as bed_file:
            # XXX: add in browser track line (see SVN revisions
            # previous to 195)
            print >>bed_file, self.make_wig_header_viterbi()

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

    def posterior2wig(self):
        infilenames = self.posterior_filenames

        range_num_segs = xrange(self.num_segs)
        wig_filenames = map(self.make_posterior_wig_filename, range_num_segs)

        zipper = izip(infilenames, self.chunk_coords)

        wig_files_unentered = [gzip_open(wig_filename, "w")
                               for wig_filename in wig_filenames]

        with nested(*wig_files_unentered) as wig_files:
            for seg_index, wig_file in enumerate(wig_files):
                print >>wig_file, self.make_wig_header_posterior(seg_index)

            for infilename, chunk_coord in zipper:
                load_posterior_write_wig(chunk_coord, self.resolution,
                                         self.num_segs, infilename, wig_files)

        # delete original input files because they are enormous
        for infilename in infilenames:
            path(infilename).remove()

    def prog_factory(self, prog):
        """
        allows dry_run
        """
        # XXX: this poisons a global variable
        prog.dry_run = self.dry_run

        return prog

    def make_acc_filename(self, thread_index, chunk_index):
        return self.make_filename(PREFIX_ACC, thread_index, chunk_index,
                                  EXT_BIN, subdirname=SUBDIRNAME_ACC)

    def make_posterior_filename(self, chunk_index):
        return self.make_filename(PREFIX_POSTERIOR, chunk_index, EXT_TXT,
                                  subdirname=SUBDIRNAME_POSTERIOR)

    def make_job_name_train(self, thread_index, round_index, chunk_index):
        return "%s%d.%d.%s.%s.%s" % (PREFIX_JOB_NAME_TRAIN, thread_index,
                                     round_index, chunk_index,
                                     self.dirpath.name, UUID)

    def make_job_name_identify(self, prefix, chunk_index):
        return "%s%d.%s.%s" % (prefix, chunk_index, self.dirpath.name,
                               UUID)

    def make_gmtk_kwargs(self):
        """
        universal args to gmtkEMtrainNew, gmtkViterbi, gmtkJT
        """
        res = dict(strFile=self.structure_filename,
                   verbosity=self.verbosity,
                   island=ISLAND,
                   componentCache=COMPONENT_CACHE,
                   deterministicChildrenStore=DETERMINISTIC_CHILDREN_STORE,
                   jtFile=self.jt_info_filename)

        if ISLAND:
            res["base"] = ISLAND_BASE
            res["lst"] = ISLAND_LST

        if HASH_LOAD_FACTOR is not None:
            res["hashLoadFactor"] = HASH_LOAD_FACTOR

        assert self.int_filelistpath
        if self.int_filelistpath:
            res.update(of1=self.int_filelistpath,
                       fmt1="binary",
                       nf1=0,
                       ni1=self.num_int_cols,
                       iswp1=SWAP_ENDIAN)

        if self.float_filelistpath:
            res.update(of2=self.float_filelistpath,
                       fmt2="binary",
                       nf2=self.num_tracks,
                       ni2=0,
                       iswp2=SWAP_ENDIAN)

        return res

    def make_chunk_lens(self):
        self.chunk_lens = [end - start
                           for chr, start, end in self.chunk_coords]

    def chunk_lens_sorted(self, reverse=True):
        """
        yields (chunk_index, chunk_mem_usage)

        if reverse: sort chunks by decreasing size, so the most
        difficult chunks are dropped in the queue first
        """
        chunk_lens = self.chunk_lens

        # XXX: use key=itemgetter(2) and enumerate instead of this silliness
        zipper = sorted(izip(chunk_lens, count()), reverse=reverse)

        # XXX: use itertools instead of a generator
        for chunk_len, chunk_index in zipper:
            yield chunk_index, chunk_len

    def calc_bayesian_info_criterion(self, log_likelihood):
        """
        BIC = -2 ln L + k ln n
        this is a modified BIC = -(2/n) ln L + k ln N

        k: # of free params
        n: # of bases
        N: # of sequences
        """
        model_penalty = (self.num_free_params * log(self.num_chunks))
        #print >>sys.stderr, "num_free_params = %s; num_bases = %s; model_penalty = %s" \
        #    % (self.num_free_params, self.num_bases, model_penalty)

        return model_penalty - (2/self.num_bases * log_likelihood)

    def log_cmdline(self, cmdline, args=None):
        if args is None:
            args = cmdline

        _log_cmdline(self.cmdline_short_file, cmdline)
        _log_cmdline(self.cmdline_long_file, args)

    def queue_gmtk(self, prog, kwargs, job_name, num_frames,
                   output_filename=None, prefix_args=[]):
        gmtk_cmdline = prog.build_cmdline(options=kwargs)
        wrapper_cmdline = [BASH_CMD, data_filename(RES_WRAPPER)]

        if prefix_args:
            # remove the command name itself from the passed arguments
            # XXX: this is ugly
            args = wrapper_cmdline + prefix_args + gmtk_cmdline[1:]
        else:
            args = wrapper_cmdline + gmtk_cmdline

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

        job_tmpl_factory = JobTemplateFactory(job_tmpl,
                                              self.mem_usage_progression,
                                              output_filename,
                                              error_filename)

        mem_usage_key = (prog.prog, self.num_segs, num_frames)

        return RestartableJob(session, job_tmpl_factory, self.global_mem_usage,
                              mem_usage_key)

    def queue_train(self, thread_index, round_index, chunk_index, num_frames=0,
                    **kwargs):
        """
        this calls Runner.queue_gmtk()

        if num_frames is not specified, then it is set to 0, where
        everyone will share their min/max memory usage. Used for calls from queue_train_bundle()
        """
        kwargs["inputMasterFile"] = self.input_master_filename

        prog = self.train_prog
        name = self.make_job_name_train(thread_index, round_index, chunk_index)

        return self.queue_gmtk(prog, kwargs, name, num_frames)

    def queue_train_parallel(self, input_params_filename, thread_index,
                             round_index, **kwargs):
        kwargs["cppCommandOptions"] = self.make_cpp_options(input_params_filename)

        res = RestartableJobDict(self.session, self.job_log_file)

        make_acc_filename_custom = partial(self.make_acc_filename, thread_index)
        num_chunks = self.num_chunks

        for chunk_index, chunk_len in self.chunk_lens_sorted():
            acc_filename = make_acc_filename_custom(chunk_index)
            kwargs_chunk = dict(trrng=chunk_index, storeAccFile=acc_filename,
                                **kwargs)

            # -dirichletPriors T only on the first chunk
            kwargs_chunk["dirichletPriors"] = (chunk_index == 0)

            num_frames = self.chunk_lens[chunk_index]

            restartable_job = self.queue_train(thread_index, round_index,
                                               chunk_index, num_frames,
                                               **kwargs_chunk)
            res.queue(restartable_job)

        return res

    def queue_train_bundle(self, input_params_filename, output_params_filename,
                           thread_index, round_index, **kwargs):
        """bundle step: take parallel accumulators and combine them
        """
        acc_filename = self.make_acc_filename(thread_index,
                                              GMTK_INDEX_PLACEHOLDER)

        cpp_options = self.make_cpp_options(input_params_filename,
                                       output_params_filename)

        last_chunk = self.num_chunks - 1

        kwargs = dict(outputMasterFile=self.output_master_filename,
                      cppCommandOptions=cpp_options,
                      trrng="nil",
                      loadAccRange="0:%s" % last_chunk,
                      loadAccFile=acc_filename,
                      **kwargs)

        restartable_job = self.queue_train(thread_index, round_index,
                                           NAME_BUNDLE_PLACEHOLDER, **kwargs)

        res = RestartableJobDict(self.session, self.job_log_file)
        res.queue(restartable_job)

        return res

    def save_posterior_triangulation(self):
        infilename = self.triangulation_filename

        # either strip ".trifile" off end, or just use the whole filename
        infilename_stem = (infilename.rpartition(SUFFIX_TRIFILE)[0]
                           or infilename)

        outfilename = extjoin(infilename_stem, EXT_POSTERIOR, EXT_TRIFILE)

        clique_indices = self.posterior_clique_indices

        # XXX: this is a fairly hacky way of doing it and will not
        # work if the triangulation file changes from what GMTK
        # generates. probably need to key on tokens rather than lines
        with open(infilename) as infile:
            with open(outfilename, "w") as outfile:
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

        self.posterior_triangulation_filename = outfilename

    def get_posterior_clique_print_ranges(self):
        res = {}

        for clique, clique_index in self.posterior_clique_indices.iteritems():
            range_str = "%d:%d" % (clique_index, clique_index)
            res[clique + "CliquePrintRange"] = range_str

        return res

    def set_triangulation_filename(self, num_segs=None):
        if num_segs is None:
            num_segs = self.num_segs

        if (self.triangulation_filename_is_new
            or not self.triangulation_filename):
            self.triangulation_filename_is_new = True

            structure_filebasename = path(self.structure_filename).name
            triangulation_filebasename = extjoin(structure_filebasename,
                                                 str(num_segs), EXT_TRIFILE)

            self.triangulation_filename = (self.triangulation_dirpath
                                           / triangulation_filebasename)

        # print >>sys.stderr, ("setting triangulation_filename = %s"
        #                     % self.triangulation_filename)

    def run_triangulate_single(self, num_segs):
        # print >>sys.stderr, "running triangulation"
        prog = self.prog_factory(TRIANGULATE_PROG)

        self.set_triangulation_filename(num_segs)

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
        self.make_triangulation_dirpath()

        num_segs_range = slice2range(self.num_segs)
        for num_segs in num_segs_range:
            self.run_triangulate_single(num_segs)

    def run_train_round(self, thread_index, round_index, **kwargs):
        """
        returns None: normal
        returns not None: abort
        """
        last_params_filename = self.last_params_filename
        curr_params_filename = extjoin(self.params_filename, str(round_index))

        restartable_jobs = \
            self.queue_train_parallel(last_params_filename, thread_index,
                                      round_index, **kwargs)
        restartable_jobs.wait()

        restartable_jobs = \
            self.queue_train_bundle(last_params_filename, curr_params_filename,
                                    thread_index, round_index,
                                    llStoreFile=self.log_likelihood_filename,
                                    **kwargs)
        restartable_jobs.wait()

        self.last_params_filename = curr_params_filename

    def set_calc_info_criterion(self):
        if self.num_free_params is None:
            # IC = -L
            self.calc_info_criterion = neg
        else:
            # IC = BIC
            self.calc_info_criterion = self.calc_bayesian_info_criterion

    def run_train_start(self):
        # make new files if you have more than one random start
        self.set_triangulation_filename()

        new = self.make_new_params

        thread_index = self.thread_index

        self.save_input_master(thread_index, new)
        self.set_params_filename(thread_index, new)
        self.set_log_likelihood_filename(thread_index, new)
        self.set_output_dirpaths(thread_index)

        last_log_likelihood = NINF
        log_likelihood = NINF
        round_index = 0

        self.set_calc_info_criterion()

        kwargs = dict(objsNotToTrain=self.dont_train_filename,
                      maxEmIters=1,
                      lldp=LOG_LIKELIHOOD_DIFF_FRAC*100.0,
                      triFile=self.triangulation_filename,
                      **self.make_gmtk_kwargs())

        while (round_index < self.max_em_iters and
               is_training_progressing(last_log_likelihood, log_likelihood)):
            self.run_train_round(thread_index, round_index, **kwargs)

            if self.dry_run:
                return (None, None, None, None)

            last_log_likelihood = log_likelihood
            log_likelihood, info_criterion = self.load_log_likelihood()

            # print >>sys.stderr, "log likelihood = %s" % log_likelihood
            # print >>sys.stderr, "info criterion = %s" % info_criterion

            round_index += 1

        # log_likelihood, num_segs and a list of src_filenames to save
        return (info_criterion, self.num_segs, self.input_master_filename,
                self.last_params_filename, self.log_likelihood_filename)

    def run_train(self):
        self.train_prog = self.prog_factory(EM_TRAIN_PROG)

        # save the destination file for input_master as we will be
        # generating new input masters for each start
        #
        # len(dst_filenames) == len(TRAIN_ATTRNAMES) == len(return value
        # of Runner.run_train_start())-1. This is asserted below.

        random_starts = self.random_starts
        assert random_starts >= 1

        # XXX: duplicative
        params_dirpath = self.dirpath / SUBDIRNAME_PARAMS

        # must be before file creation. Otherwise the newness value
        # will be wrong
        input_master_filename, input_master_filename_is_new = \
            make_template_filename(self.input_master_filename,
                                   RES_INPUT_MASTER_TMPL, params_dirpath,
                                   self.clobber)

        # should I make new parameters in each thread?
        make_new_params = (self.random_starts > 1
                           or isinstance(self.num_segs, slice))
        self.make_new_params = make_new_params
        if not make_new_params:
            self.save_input_master()

        if not input_master_filename_is_new:
            # do not overwrite existing file
            input_master_filename = None

        dst_filenames = [input_master_filename,
                         self.params_filename,
                         self.log_likelihood_filename]

        num_segs_range = slice2range(self.num_segs)

        if len(num_segs_range) > 1 or random_starts > 1:
            thread_runner = self.run_train_multithread
        else:
            thread_runner = self.run_train_singlethread

        start_params = thread_runner(num_segs_range)

        if self.make_new_params:
            self.proc_train_results(start_params, dst_filenames)
        else:
            # only one random start: always overwrite params.params
            assert len(start_params) == 1
            copy2(start_params[0][OFFSET_PARAMS_FILENAME],
                  self.params_filename)

    def run_train_singlethread(self, num_segs_range):
        # having a single-threaded version makes debugging much easier
        with Session() as session:
            self.session = session
            self.thread_index = 0
            start_params = [self.run_train_start()]

        self.session = None

        return start_params

    def run_train_multithread(self, num_segs_range):
        # XXX: Python 2.6 use itertools.product()
        seg_thread_indexes = xrange(self.random_starts)
        enumerator = enumerate((num_seg, seg_thread_index)
                               for num_seg in num_segs_range
                               for seg_thread_index in seg_thread_indexes)

        threads = []
        with Session() as session:
            try:
                for thread_index, (num_seg, seg_thread_index) in enumerator:
                    # print >>sys.stderr, (
                    #    "thread_index %s, num_seg %s, seg_thread_index %s"
                    #    % (thread_index, num_seg, seg_thread_index))
                    thread = RandomStartThread(self, session, thread_index,
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
                start_params = []
                for thread in threads:
                    while thread.isAlive():
                        # XXX: KeyboardInterrupts only occur if there is a
                        # timeout specified here. Is this a Python bug?
                        thread.join(JOIN_TIMEOUT)

                    # this will get AttributeError if the thread failed and
                    # therefore did not set thread.result
                    start_params.append(thread.result)
            except KeyboardInterrupt:
                self.interrupt_event.set()
                for thread in threads:
                    thread.join()

                raise

        return start_params

    def proc_train_results(self, start_params, dst_filenames):
        if self.dry_run:
            return

        # finds the min by info_criterion (minimize -log_likelihood)
        min_params = min(start_params)

        self.num_segs = min_params[OFFSET_NUM_SEGS]
        self.set_triangulation_filename()

        src_filenames = min_params[OFFSET_FILENAMES:]

        if None in src_filenames:
            raise ValueError("all training threads failed")

        assert LEN_TRAIN_ATTRNAMES == len(src_filenames) == len(dst_filenames)

        zipper = zip(TRAIN_ATTRNAMES, src_filenames, dst_filenames)
        for name, src_filename, dst_filename in zipper:
            self.copy_results(name, src_filename, dst_filename)

    def _queue_identify(self, restartable_jobs, chunk_index, prefix_job_name,
                        prog, kwargs, output_filenames):
        prog = self.prog_factory(prog)
        job_name = self.make_job_name_identify(prefix_job_name, chunk_index)
        output_filename = output_filenames[chunk_index]

        kwargs = self.get_identify_kwargs(chunk_index, kwargs)

        if prog == VITERBI_PROG:
            chunk_coord = self.chunk_coords[chunk_index]
            chunk_chrom, chunk_start, chunk_end = chunk_coord

            float_filepath, int_filepath = \
                self.make_obs_filepaths(chunk_chrom, chunk_index, temp=True)

            prefix_args = [find_executable("segway-task"), "run", "viterbi",
                           output_filename, chunk_chrom, chunk_start,
                           chunk_end, self.resolution, self.num_segs,
                           self.genomedata_dirname, float_filepath,
                           int_filepath, self.distribution,
                           self.track_indexes_text]
            output_filename = None
        else:
            prefix_args = []

        num_frames = self.chunk_lens[chunk_index]

        restartable_job = self.queue_gmtk(prog, kwargs, job_name,
                                          num_frames,
                                          output_filename=output_filename,
                                          prefix_args=prefix_args)

        restartable_jobs.queue(restartable_job)

    def get_identify_kwargs(self, chunk_index, extra_kwargs):
        cpp_command_options = self.make_cpp_options(self.params_filename)

        res = dict(inputMasterFile=self.input_master_filename,
                   cppCommandOptions=cpp_command_options,
                   dcdrng=chunk_index,
                   cliqueTableNormalize="0.0",
                   **self.make_gmtk_kwargs())

        res.update(extra_kwargs)

        return res

    def run_identify_posterior(self, clobber=None):
        ## setup files
        if not self.input_master_filename:
            self.save_input_master()

        self.set_output_dirpaths("identify", clobber)
        self.make_posterior_filenames()

        viterbi_filenames = self.viterbi_filenames
        posterior_filenames = self.posterior_filenames

        # -: standard output, processed by segway-task
        viterbi_kwargs = dict(triFile=self.triangulation_filename,
                              pVitRegexFilter="^seg$",
                              pVitValsFile="-")

        posterior_kwargs = dict(triFile=self.posterior_triangulation_filename,
                                doDistributeEvidence=True,
                                **self.get_posterior_clique_print_ranges())

        # XXX: kill submitted jobs on exception
        jobids = []
        with Session() as session:
            self.session = session

            restartable_jobs = RestartableJobDict(session, self.job_log_file)

            for chunk_index, chunk_len in self.chunk_lens_sorted():
                queue_identify_custom = partial(self._queue_identify,
                                                restartable_jobs, chunk_index)

                if self.identify:
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

        # XXXopt: parallelize
        if self.posterior:
            self.posterior2wig()

    def run(self):
        """
        main run, after dirname is specified

        this is exposed so that it can be overriden in a subclass
        """
        # XXXopt: use binary I/O to gmtk rather than ascii for parameters

        self.dirpath = path(self.dirname)

        self.make_subdir(SUBDIRNAME_LOG)
        cmdline_short_filename = self.make_filename(PREFIX_CMDLINE_SHORT,
                                                    EXT_SH,
                                                    subdirname=SUBDIRNAME_LOG)
        cmdline_long_filename = self.make_filename(PREFIX_CMDLINE_LONG, EXT_SH,
                                                    subdirname=SUBDIRNAME_LOG)
        cmdline_top_filename = self.make_filename(PREFIX_CMDLINE_TOP, EXT_SH,
                                                  subdirname=SUBDIRNAME_LOG)
        job_log_filename = self.make_filename(PREFIX_JOB_LOG, EXT_TAB,
                                                subdirname=SUBDIRNAME_LOG)

        self.interrupt_event = Event()

        self.save_params()

        now = datetime.now()
        pkg_desc = working_set.find(Requirement.parse(PKG))
        run_msg = "## %s run %s at %s" % (pkg_desc, UUID, now)

        with open(cmdline_top_filename, "w") as cmdline_top_file:
            print >>cmdline_top_file, run_msg
            print >>cmdline_top_file
            print >>cmdline_top_file, cmdline2text()

        with open(cmdline_short_filename, "w") as cmdline_short_file:
            # XXX: works around a pyflakes bug for with * as self.*; report
            self.cmdline_short_file = cmdline_short_file

            with open(cmdline_long_filename, "w") as cmdline_long_file:
                self.cmdline_long_file = cmdline_long_file

                print >>cmdline_short_file, run_msg
                print >>cmdline_long_file, run_msg

                # so that we can immediately get the UUID if we want it
                self.cmdline_short_file.flush()

                if self.triangulate:
                    self.run_triangulate()

                with open(job_log_filename, "w") as job_log_file:
                    self.job_log_file = job_log_file
                    print >>job_log_file, "\t".join(JOB_LOG_FIELDNAMES)

                    if self.train:
                        self.run_train()

                    if self.identify or self.posterior:
                        if not self.dry_run:
                            # resave now that num_segs is determined
                            self.save_include()

                            if not self.posterior_triangulation_filename:
                                self.save_posterior_triangulation()

                        try:
                            self.run_identify_posterior()
                        except ChunkOverMemUsageLimit:
                            self.resave_params()

                            # erase old output
                            self.run_identify_posterior(clobber=True)

    def __call__(self, *args, **kwargs):
        # XXX: register atexit for cleanup_resources

        dirname = self.dirname
        if dirname:
            if self.clobber or not path(dirname).isdir():
                self.make_dir(dirname)

            self.run(*args, **kwargs)
        else:
            try:
                with NamedTemporaryDir(prefix=TEMPDIR_PREFIX) as tempdir:
                    self.dirname = tempdir.name
                    self.is_dirname_temp = True
                    self.run()
            finally:
                # the temporary directory has already been deleted (after
                # unwinding of the with block), so let's stop referring to
                # it
                self.dirname = None
                self.is_dirname_temp = False

def parse_options(args):
    from optplus import OptionParser, OptionGroup

    usage = "%prog [OPTION]... GENOMEDATADIR"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    with OptionGroup(parser, "Input selection") as group:
        group.add_option("-t", "--track", action="append", default=[],
                         metavar="TRACK",
                         help="append TRACK to list of tracks to use"
                         " (default all)")

        # This is a 0-based file.
        # I know because ENm008 starts at position 0 in encodeRegions.txt.gz
        group.add_option("--include-coords", metavar="FILE",
                         help="limit to genomic coordinates in FILE")

        # exclude goes after all includes
        group.add_option("--exclude-coords", metavar="FILE",
                         help="filter out genomic coordinates in FILE")

    with OptionGroup(parser, "Model files") as group:
        group.add_option("-i", "--input-master", metavar="FILE",
                         help="use or create input master in FILE")

        group.add_option("-s", "--structure", metavar="FILE",
                         help="use or create structure in FILE")

        group.add_option("-p", "--trainable-params", action="append",
                         default=[], metavar="FILE",
                         help="use or create trainable parameters in FILE")

        group.add_option("--dont-train", metavar="FILE",
                         help="use FILE as list of parameters not to train")

        group.add_option("--seg-table", metavar="FILE",
                         help="load segment hyperparameters from FILE")

        group.add_option("--semisupervised", metavar="FILE",
                         help="semisupervised segmentation with labels in "
                         "FILE")

    with OptionGroup(parser, "Output files") as group:
        group.add_option("-b", "--bed", metavar="FILE",
                          help="create bed track in FILE")

    with OptionGroup(parser, "Intermediate files") as group:
        # XXX: consider removing this option
        # this probably isn't necessary as observations are written
        # out pretty quickly now
        group.add_option("-o", "--observations", metavar="DIR",
                          help="use or create observations in DIR")

        group.add_option("-d", "--directory", metavar="DIR",
                          help="create all other files in DIR")

    with OptionGroup(parser, "Variables") as group:
        group.add_option("-D", "--distribution", choices=DISTRIBUTIONS,
                         metavar="DIST", default=DISTRIBUTION_DEFAULT,
                         help="use DIST distribution")

        group.add_option("-r", "--random-starts", type=int,
                         default=RANDOM_STARTS, metavar="NUM",
                         help="randomize start parameters NUM times"
                         " (default %d)" % RANDOM_STARTS)

        group.add_option("-N", "--num-labels", type=slice,
                         default=NUM_SEGS, metavar="SLICE",
                         help="make SLICE segment labels"
                         " (default %d)" % NUM_SEGS)

        group.add_option("--resolution", type=int,
                         default=RESOLUTION, metavar="RES",
                         help="downsample to every RES bp (default %d)" %
                         RESOLUTION)

        group.add_option("--ruler-scale", type=int,
                         default=RULER_SCALE, metavar="SCALE",
                         help="ruler marking every SCALE bp (default %d)" %
                         RULER_SCALE)

        group.add_option("--prior-strength", type=float,
                         default=PRIOR_STRENGTH, metavar="RATIO",
                         help="use RATIO times the number of data counts as"
                         " the number of pseudocounts for the segment length"
                         " prior (default %d)" % PRIOR_STRENGTH)

        group.add_option("-m", "--mem-usage", default=MEM_USAGE_PROGRESSION,
                         metavar="PROGRESSION",
                         help="try each float in PROGRESSION as the number "
                         "of gibibytes of memory to allocate in turn "
                         "(default %s)" % MEM_USAGE_PROGRESSION)

        group.add_option("-S", "--split-sequences", metavar="SIZE",
                         default=MAX_FRAMES,
                         help="split up sequences that are larger than SIZE "
                         "bp (default %s)" % MAX_FRAMES)

        group.add_option("-v", "--verbosity", type=int, default=VERBOSITY,
                         metavar="NUM",
                         help="show messages with verbosity NUM")

        group.add_option("--cluster-opt", action="append", default=[],
                         metavar="OPT",
                         help="specify an option to be passed to the "
                         "cluster manager")

    with OptionGroup(parser, "Flags") as group:
        group.add_option("-c", "--clobber", action="store_true",
                         help="delete any preexisting files")
        group.add_option("-T", "--no-train", action="store_true",
                         help="do not train model")
        group.add_option("-I", "--no-identify", action="store_true",
                         help="do not identify segments")
        group.add_option("-P", "--no-posterior", action="store_true",
                         help="do not identify probability of segments")
        group.add_option("-k", "--keep-going", action="store_true",
                         help="keep going in some threads even when you have"
                         " errors in another")
        group.add_option("-n", "--dry-run", action="store_true",
                         help="write all files, but do not run any"
                         " executables")

    options, args = parser.parse_args(args)

    if not len(args) == 1:
        parser.error("Expected only one argument.")

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    runner = Runner.fromoptions(args, options)

    return runner()

if __name__ == "__main__":
    sys.exit(main())
