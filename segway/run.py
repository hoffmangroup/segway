#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008-2009 Michael M. Hoffman <mmh1@washington.edu>

from cStringIO import StringIO
from collections import defaultdict
from contextlib import closing, nested
from copy import copy
from errno import EEXIST, ENOENT
from functools import partial
from itertools import count, izip
from math import ceil, floor, frexp, ldexp, log, log10
from os import extsep
import re
from shutil import move
from string import Template
import sys
from threading import Event, Thread
from uuid import uuid1

from DRMAA import ExitTimeoutError
from numpy import (add, amin, amax, append, arange, array, column_stack,
                   diagflat, empty, finfo, float32, fromfile, intc, invert,
                   isnan, NINF, ones, outer, s_, sqrt, square, tile,
                   vectorize, zeros)
from numpy.random import uniform
from optbuild import (Mixin_NoConvertUnderscore,
                      Mixin_UseFullProgPath,
                      OptionBuilder_ShortOptWithSpace,
                      OptionBuilder_ShortOptWithSpace_TF)
from path import path
from tabdelim import DictReader, ListWriter

from .res_usage import convert_sge_mem_size, fetch_sge_qacct_records
from ._util import (data_filename, data_string, DTYPE_IDENTIFY, DTYPE_OBS_INT,
                    EXT_GZ, find_segment_starts, get_tracknames,
                    gzip_open, ISLAND_BASE_NA, ISLAND_LST_NA,
                    iter_chroms_coords, load_coords, NamedTemporaryDir, PKG,
                    Session, walk_continuous_supercontigs, walk_supercontigs)

# XXX: I should really get some sort of Enum for this, I think Peter
# Norvig has one
DISTRIBUTION_NORM = "norm"
DISTRIBUTION_GAMMA = "gamma"
DISTRIBUTIONS = [DISTRIBUTION_NORM, DISTRIBUTION_GAMMA]
DISTRIBUTION_DEFAULT = DISTRIBUTION_NORM

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

MIN_NUM_SEGS = 2
NUM_SEGS = MIN_NUM_SEGS
MIN_SEG_LEN = 1
MAX_EM_ITERS = 100
TEMPDIR_PREFIX = PKG + "-"
COVAR_TIED = True # would need to expand to MC, MX to change
MAX_CHUNKS = 1000

ISLAND = True

# XXX: temporary code to allow easy switching
if ISLAND:
    ISLAND_BASE = 3
    ISLAND_LST = 100000
else:
    ISLAND_BASE = ISLAND_BASE_NA
    ISLAND_LST = ISLAND_LST_NA

ISLAND = ISLAND_BASE != ISLAND_BASE_NA
assert ISLAND or ISLAND_LST == ISLAND_LST_NA

SESSION_WAIT_TIMEOUT = 60 # seconds
JOIN_TIMEOUT = finfo(float).max
LEN_SEG_EXPECTED = 10000
SWAP_ENDIAN = False
POSTERIOR_SCALE_FACTOR = 100.0

## option defaults
VERBOSITY = 0
PRIOR_STRENGTH = 0

FINFO_FLOAT32 = finfo(float32)
MACHEP_FLOAT32 = FINFO_FLOAT32.machep
TINY_FLOAT32 = FINFO_FLOAT32.tiny

JITTER_STD_BOUND = 0.2
FUDGE_EP = -17 # ldexp(1, -17) = ~1e-6
assert FUDGE_EP > MACHEP_FLOAT32

# binary
JITTER_ORDERS_MAGNITUDE = 5 # log10(2**5) = 1.5 decimal orders of magnitude

FUDGE_TINY = -ldexp(TINY_FLOAT32, 6)
ABSOLUTE_FUDGE = 0.001

LOG_LIKELIHOOD_DIFF_FRAC = 1e-5

ORD_A = ord("A")
ORD_C = ord("C")
ORD_G = ord("G")
ORD_T = ord("T")
ORD_a = ord("a")
ORD_c = ord("c")
ORD_g = ord("g")
ORD_t = ord("t")
NUM_SEQ_COLS = 2 # dinucleotide, presence_dinucleotide

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2
MAX_FRAMES = 1000000000 # 1 billion
MEM_USAGE_LIMIT = 15*2**30 # 15 GB
MEM_USAGE_BUNDLE = 100000000 # 100M; XXX: should be included in calibration
RES_REQ_IDS = ["mem_requested", "h_vmem"] # h_vmem: hard ulimit
ALWAYS_MAX_MEM_USAGE = True

POSTERIOR_CLIQUE_INDICES = dict(p=1, c=1, e=1)

## defaults
RANDOM_STARTS = 1

# replace NAN with SENTINEL to avoid warnings
# XXX: replace with something negative and outlandish again
SENTINEL = float32(9.87654321)

CPP_DIRECTIVE_FMT = "-D%s=%s"

GMTK_INDEX_PLACEHOLDER = "@D"
NAME_BUNDLE_PLACEHOLDER = "bundle"

# programs
ENV_CMD = "/usr/bin/env"
BASH_CMD = "bash"

BASH_CMDLINE = [BASH_CMD, "--login", "-c"]

OptionBuilder_GMTK = (Mixin_UseFullProgPath +
                      OptionBuilder_ShortOptWithSpace_TF)

TRIANGULATE_PROG = OptionBuilder_GMTK("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_GMTK("gmtkEMtrainNew")
VITERBI_PROG = OptionBuilder_GMTK("gmtkViterbi")
POSTERIOR_PROG = OptionBuilder_GMTK("gmtkJT")

NATIVE_SPEC_PROG = (Mixin_NoConvertUnderscore
                    + OptionBuilder_ShortOptWithSpace)() # do not run

NATIVE_SPEC_DEFAULT = dict(w="n")

SPECIAL_TRACKNAMES = ["dinucleotide"]

def extjoin(*args):
    return extsep.join(args)

# extensions and suffixes
EXT_BED = "bed"
EXT_BIN = "bin"
EXT_IDENTIFY = "identify.h5"
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
PREFIX_CMDLINE = "run"
PREFIX_POSTERIOR = "posterior"

PREFIX_VITERBI = "viterbi"
PREFIX_LIKELIHOOD = "likelihood"
PREFIX_CHUNK = "chunk"
PREFIX_PARAMS = "params"
PREFIX_JT_INFO = "jt_info"

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
SUBDIRNAME_PARAMS = "params" # XXX: final params should go into main directory
SUBDIRNAME_POSTERIOR = "posterior"
SUBDIRNAME_VITERBI = "viterbi"

SUBDIRNAMES_EITHER = [SUBDIRNAME_AUX]
SUBDIRNAMES_TRAIN = [SUBDIRNAME_ACC, SUBDIRNAME_LIKELIHOOD,
                     SUBDIRNAME_PARAMS]
SUBDIRNAMES_IDENTIFY = [SUBDIRNAME_POSTERIOR, SUBDIRNAME_VITERBI]

FLOAT_TAB_FIELDNAMES = ["filename", "chunk_index", "chrom", "start", "end"]

# templates and formats
RES_STR_TMPL = "segway.str.tmpl"
RES_INPUT_MASTER_TMPL = "input.master.tmpl"
RES_OUTPUT_MASTER = "output.master"
RES_DONT_TRAIN = "dont_train.list"
RES_INC_TMPL = "segway.inc.tmpl"
RES_DUMPNAMES = "dumpnames.list" # XXX: remove all dumpnames stuff from code
RES_RES_USAGE = "res_usage.tab"

DIRICHLET_FRAG = "0 dirichlet_state_state 2 CARD_STATE CARD_STATE"

# XXX: manually indexing these things is silly
DENSE_CPT_START_STATE_FRAG = "0 start_state 0 CARD_STATE"
DENSE_CPT_STATE_STATE_FRAG = "1 state_state 1 CARD_STATE CARD_STATE"
DIRICHLET_STATE_STATE_FRAG = "DirichletTable dirichlet_state_state"

DENSE_CPT_SEG_DINUCLEOTIDE_FRAG = \
    "2 seg_dinucleotide 1 CARD_SEG CARD_DINUCLEOTIDE"

MEAN_TMPL = "$index mean_${seg}_${track} 1 ${rand}"

COVAR_TMPL_TIED = "$index covar_${track} 1 ${rand}"
# XXX: unused
COVAR_TMPL_UNTIED = "$index covar_${seg}_${track} 1 ${rand}"

GAMMASCALE_TMPL = "$index gammascale_${seg}_${track} 1 1 ${rand}"
GAMMASHAPE_TMPL = "$index gammashape_${seg}_${track} 1 1 ${rand}"

MC_NORM_TMPL = "$index 1 COMPONENT_TYPE_DIAG_GAUSSIAN" \
    " mc_norm_${seg}_${track} mean_${seg}_${track} covar_${track}"
MC_GAMMA_TMPL = "$index 1 COMPONENT_TYPE_GAMMA mc_gamma_${seg}_${track}" \
    " ${min_track} gammascale_${seg}_${track} gammashape_${seg}_${track}"
MC_TMPLS = dict(norm=MC_NORM_TMPL,
                gamma=MC_GAMMA_TMPL)

MX_TMPL = "$index 1 mx_${seg}_${track} 1 dpmf_always" \
    " mc_${distribution}_${seg}_${track}"

NAME_COLLECTION_TMPL = "$track_index collection_seg_${track} 2"
NAME_COLLECTION_CONTENTS_TMPL = "mx_${seg}_${track}"

TRACK_FMT = "browser position %s:%s-%s"
FIXEDSTEP_FMT = "fixedStep chrom=%s start=%s step=1 span=1"

WIG_ATTRS = dict(type="wiggle_0",
                 autoScale="off")
WIG_ATTRS_VITERBI = dict(name="%s" % PKG,
                         visibility="dense",
                         viewLimits="0:1",
                         **WIG_ATTRS)
WIG_ATTRS_POSTERIOR = dict(viewLimits="0:100",
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

FIXEDSTEP_HEADER = "fixedStep chrom=%s start=%s step=1"

MSG_SUCCESS = "____ PROGRAM ENDED SUCCESSFULLY WITH STATUS 0 AT"

# set once per file run
UUID = uuid1().hex

## functions
def _constant(constant):
    """
    constant values for defaultdict
    """
    def _constant_inner():
        return constant

    return _constant_inner

def make_fixedstep_header(chrom, start):
    """
    this function expects 0-based coordinates
    it does the conversion to 1-based coordinates for you!
    """
    start_1based = start+1
    return FIXEDSTEP_HEADER % (chrom, start_1based)

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

class NoAdvance(str):
    """
    cause rewrite_strip_comments() to not consume an extra line
    """

class NewLine(NoAdvance):
    """
    add a line rather than replacing existing
    """
    # doesn't actually have any code. used slowly for class identification

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

def make_res_req(size):
    res = []
    for res_req_id in RES_REQ_IDS:
        res.append("%s=%s" % (res_req_id, size))

    return res

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

def save_template(filename, resource, mapping, dirname=None,
                  clobber=False, start_index=None):
    """
    creates a temporary file if filename is None or empty
    """
    if filename:
        if not clobber and path(filename).exists():
            return filename, False
    else:
        resource_part = resource.rpartition(".tmpl")
        stem = resource_part[0] or resource_part[2]
        stem_part = stem.rpartition(extsep)
        prefix = stem_part[0]
        ext = stem_part[2]

        filebasename = extjoin_not_none(prefix, start_index, ext)

        filename = path(dirname) / filebasename

    with open(filename, "w+") as outfile:
        tmpl = Template(data_string(resource))
        text = tmpl.substitute(mapping)

        outfile.write(text)

    return filename, True

def find_overlaps(start, end, coords):
    """
    find items in coords that overlap (start, end)

    NOTE: multiple overlapping regions in coords will result in data
    being considered more than once
    """
    res = []

    for coord_start, coord_end in coords:
        if start > coord_end:
            pass
        elif end <= coord_start:
            pass
        elif start <= coord_start:
            if end >= coord_end:
                res.append([coord_start, coord_end])
            else:
                res.append([coord_start, end])
        elif start > coord_start:
            if end >= coord_end:
                res.append([start, coord_end])
            else:
                res.append([start, end])

    return res

def make_cpp_options(input_params_filename, output_params_filename=None):
    directives = {}

    if input_params_filename:
        directives["INPUT_PARAMS_FILENAME"] = input_params_filename

    if output_params_filename:
        directives["OUTPUT_PARAMS_FILENAME"] = output_params_filename

    res = " ".join(CPP_DIRECTIVE_FMT % item for item in directives.iteritems())

    if res:
        return res

    # default: return None

def make_native_spec(**kwargs):
    options = NATIVE_SPEC_DEFAULT.copy()
    options.update(kwargs)

    res = " ".join(NATIVE_SPEC_PROG.build_args(options=options))

    return res

def make_spec(name, items):
    items[:0] = ["%s_IN_FILE inline" % name, str(len(items)), ""]

    return "\n".join(items) + "\n"

def make_table_spec(frag, table):
    items = [frag] + [" ".join(map(str, row)) for row in table]
    return "\n".join(items) + "\n"

# def make_dt_spec(num_tracks):
#     return make_spec("DT", ["%d seg_obs%d BINARY_DT" % (index, index)
#                             for index in xrange(num_tracks)])

def prob_transition_from_expected_len(length):
    # formula from Meta-MEME paper, Grundy WN et al. CABIOS 13:397
    return length / (1 + length)

def make_name_collection_spec(num_segs, tracknames):
    substitute = Template(NAME_COLLECTION_TMPL).substitute
    substitute_contents = Template(NAME_COLLECTION_CONTENTS_TMPL).substitute

    items = []

    for track_index, track in enumerate(tracknames):
        mapping = dict(track=track, track_index=track_index)

        contents = [substitute(mapping)]
        for seg_index in xrange(num_segs):
            seg = "seg%d" % seg_index
            mapping = dict(seg=seg, track=track,
                           seg_index=seg_index, track_index=track_index)

            contents.append(substitute_contents(mapping))
        items.append("\n".join(contents))

    return make_spec("NAME_COLLECTION", items)

re_seg = re.compile(r"^seg\((\d+)\)=(\d+)$")
def load_gmtk_out(filename):
    with open(filename) as infile:
        infile_lines = iter(infile)

        assert infile_lines.next().startswith("Segment ")
        assert infile_lines.next().startswith("========")

        line = infile_lines.next()
        assert line.startswith("Segment ")

        num_frames_text = line.split(", ")[1].partition(" = ")
        assert num_frames_text[0] == "number of frames"
        assert num_frames_text[1] == " = "

        num_frames = int(num_frames_text[2])

        line = infile_lines.next()
        assert line.startswith("Printing random variables from (P,C,E)")

        res = zeros(num_frames, DTYPE_IDENTIFY)

        for line in infile_lines:
            if line.startswith(MSG_SUCCESS):
                return res
            assert line.rstrip().endswith(" partition")

            line = infile_lines.next()
            for pair in line.split(","):
                match = re_seg.match(pair)
                if not match:
                    continue

                index = int(match.group(0))
                val = int(match.group(1))

                res[index] = val

    raise ValueError("%s did not complete successfully" % VITERBI_PROG.prog)

def write_bed(outfile, start_pos, labels, coords):
    (chrom, region_start, region_end) = coords

    start_pos += region_start

    zipper = zip(start_pos[:-1], start_pos[1:], labels)
    for seg_start, seg_end, seg_label in zipper:
        row = [chrom, str(seg_start), str(seg_end), str(seg_label)]
        print >>outfile, "\t".join(row)

def load_gmtk_out_write_bed(coords, gmtk_outfilename, bed_file):
    data = load_gmtk_out(gmtk_outfilename)

    start_pos, labels = find_segment_starts(data)

    write_bed(bed_file, start_pos, labels, coords)

def parse_posterior(iterable):
    # skip to first frame
    for line in iterable:
        if line.startswith("-"):
            break

    # skip frame header
    iterable.next()

    res = []
    for line in iterable:
        # frame boundary or file end
        if line.startswith(("-", "_")):
            yield res
            res = []
            iterable.next() # skip frame header
        else:
            # return the first word, which is the posterior probability
            res.append(float(line.split()[1]))

def load_posterior_write_wig((chrom, start, end), infilename, outfiles):
    header = make_fixedstep_header(chrom, start)

    for outfile in outfiles:
        print >>outfile, header

    with open(infilename) as infile:
        for probs in parse_posterior(infile):
            for outfile, prob in zip(outfiles, probs):
                print >>outfile, int(round(prob * POSTERIOR_SCALE_FACTOR))

def set_cwd_job_tmpl(job_tmpl):
    job_tmpl.workingDirectory = path.getcwd()

def make_dinucleotide_int_data(seq):
    """
    makes an array with two columns, one with 0..15=AA..TT and the other
    as a presence variable. Set column one to 0 when not present
    """
    nucleotide_int_data = (((seq == ORD_A) + (seq == ORD_a))
                           + ((seq == ORD_C) + (seq == ORD_c)) * 2
                           + ((seq == ORD_G) + (seq == ORD_g)) * 3
                           + ((seq == ORD_T) + (seq == ORD_t)) * 4) - 1
    nucleotide_missing = nucleotide_int_data == -1

    # rewrite all Ns as A now that you have the missingness mask
    nucleotide_int_data[nucleotide_int_data == -1] = 0
    col_shape = (len(nucleotide_int_data)-1,)

    # first column: dinucleotide: AA..TT=0..15
    # combine, and add extra AA stub at end, which will be set missing
    # 0 AA AC AG AT
    # 4 CA CC CG CT
    # 8 GA GC GG GT
    # 12 TA TC TG TT
    dinucleotide_int_data = empty(col_shape, DTYPE_OBS_INT)
    add(nucleotide_int_data[:-1] * 4, nucleotide_int_data[1:],
        dinucleotide_int_data)

    # second column: presence_dinucleotide: some missing=0; all present = 1
    # there are so few N boundaries that it is okay to
    # disregard the whole dinucleotide when half is N
    dinucleotide_missing = (nucleotide_missing[:-1] + nucleotide_missing[1:])

    dinucleotide_presence = empty(dinucleotide_missing.shape, DTYPE_OBS_INT)
    invert(dinucleotide_missing, dinucleotide_presence)

    # XXXopt: set these up properly in the first place instead of
    # column_stacking at the end
    return column_stack([dinucleotide_int_data, dinucleotide_presence])

def walk_all_supercontigs(h5file):
    """
    shares interface with walk_continuous_supercontigs()
    """
    for supercontig in walk_supercontigs(h5file):
        yield supercontig, None

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
    return "%dG" % ceil(mem_usage / 2**30)

def update_starts(starts, ends, new_starts, new_ends, start_index):
    next_index = start_index + 1

    starts[next_index:next_index] = new_starts
    ends[next_index:next_index] = new_ends

def fetch_sge_qacct_single_record(jobname):
    records = list(fetch_sge_qacct_records(jobname))
    assert len(records) == 1

    return records[0]

def get_sge_qacct_maxvmem(jobname):
    record = fetch_sge_qacct_single_record(jobname)
    return convert_sge_mem_size(record["maxvmem"])

class RandomStartThread(Thread):
    def __init__(self, runner, session, start_index, num_segs,
                 interrupt_event):
        # keeps it from rewriting variables that will be used
        # later or in a different thread
        self.runner = copy(runner)

        self.session = session
        self.num_segs = num_segs
        self.start_index = start_index
        self.interrupt_event = interrupt_event

        Thread.__init__(self)

    def run(self):
        self.runner.session = self.session
        self.runner.num_segs = self.num_segs
        self.runner.start_index = self.start_index
        self.runner.interrupt_event = self.interrupt_event
        self.result = self.runner.run_train_start()

class Runner(object):
    def __init__(self, **kwargs):
        # filenames
        self.h5filenames = None
        self.float_filelistpath = None
        self.int_filelistpath = None

        self.gmtk_include_filename = None
        self.input_master_filename = None
        self.structure_filename = None
        self.triangulation_filename = None
        self.posterior_triangulation_filename = None
        self.jt_info_filename = None
        self.res_usage_filename = data_filename(RES_RES_USAGE)

        self.params_filename = None
        self.dirname = None
        self.is_dirname_temp = False
        self.log_likelihood_filename = None
        self.log_likelihood_log_filename = None
        self.dont_train_filename = None

        self.dumpnames_filename = None
        self.viterbi_filelistname = None # XXX: remove from code everywhere
        self.viterbi_filenames = None

        self.obs_dirname = None
        self.bed_filename = None

        self.include_coords_filename = None

        self.posterior_clique_indices = POSTERIOR_CLIQUE_INDICES.copy()

        # data
        # a "chunk" is what GMTK calls a segment
        self.num_chunks = None
        self.chunk_coords = None
        self.mins = None
        self.maxs = None
        self.tracknames = None
        self.mem_per_obs = defaultdict(_constant(None))

        # variables
        self.num_segs = NUM_SEGS
        self.random_starts = RANDOM_STARTS
        self.len_seg_strength = PRIOR_STRENGTH
        self.distribution = DISTRIBUTION_DEFAULT
        self.max_em_iters = MAX_EM_ITERS
        self.min_seg_len = MIN_SEG_LEN

        # flags
        self.clobber = False
        self.triangulate = True
        self.train = True # EM train # this should become an int for num_starts
        self.posterior = True
        self.verbosity = VERBOSITY
        self.identify = True # viterbi
        self.dry_run = False
        self.use_dinucleotide = None
        self.skip_large_mem_usage = False
        self.split_sequences = False

        # functions
        self.train_prog = None

        self.__dict__.update(kwargs)

    def set_mem_per_obs(self, jobname, chunk_len, prog):
        maxvmem = get_sge_qacct_maxvmem(jobname)

        if ISLAND:
            chunk_len = log(chunk_len)

        mem_per_obs = maxvmem / chunk_len

        print >>sys.stderr, "mem_per_obs = %d / %d = %d" % (maxvmem, chunk_len,
                                                            mem_per_obs)
        self.mem_per_obs[prog] = mem_per_obs

    def load_log_likelihood(self):
        with open(self.log_likelihood_filename) as infile:
            res = infile.read().strip()

        with open(self.log_likelihood_log_filename, "a") as logfile:
            print >>logfile, res

        return float(res)

    def load_include_coords(self):
        filename = self.include_coords_filename

        self.include_coords = load_coords(filename)

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
        tracknames = get_tracknames(chromosome)

        # includes specials
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
        elif (self.tracknames != tracknames
              or (self.track_indexes != track_indexes).any()):
            raise ValueError("all tracknames attributes must be identical")

        return track_indexes

    def set_jt_info_filename(self):
        if not self.jt_info_filename:
            self.jt_info_filename = \
                self.make_filename(PREFIX_JT_INFO, EXT_TXT,
                                   subdirname=SUBDIRNAME_LOG)

    def set_params_filename(self, start_index=None, new=False):
        # if this is not run and params_filename is
        # unspecified, then it won't be passed to gmtkViterbiNew

        params_filename = self.params_filename
        if not new and params_filename:
            if (not self.clobber
                and path(params_filename).exists()):
                # it already exists and you don't want to force regen
                self.train = False
        else:
            self.params_filename = \
                self.make_filename(PREFIX_PARAMS, start_index, EXT_PARAMS,
                                   subdirname=SUBDIRNAME_PARAMS)

    def set_log_likelihood_filename(self, start_index=None, new=False):
        if new or not self.log_likelihood_filename:
            log_likelihood_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, start_index,
                                   EXT_LIKELIHOOD,
                                   subdirname=SUBDIRNAME_LIKELIHOOD)

            self.log_likelihood_filename = log_likelihood_filename

            self.log_likelihood_log_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, start_index, EXT_TXT,
                                   subdirname=SUBDIRNAME_LOG)

    def make_output_dirpath(self, dirname, start_index):
        res = self.dirpath / "output" / dirname / str(start_index)
        self.make_dir(res)

        return res

    def set_output_dirpaths(self, start_index):
        self.output_dirpath = self.make_output_dirpath("o", start_index)
        self.error_dirpath = self.make_output_dirpath("e", start_index)

    def make_dir(self, dirname):
        dirpath = path(dirname)

        if self.clobber:
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

    def make_obs_filepath(self, prefix, suffix):
        return self.obs_dirpath / (prefix + suffix)

    def make_obs_filepaths(self, chrom, chunk_index):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_CHUNKS))
        prefix = prefix_feature_tmpl % chunk_index

        make_obs_filepath_custom = partial(self.make_obs_filepath, prefix)

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
        num_states = num_segs * self.min_seg_len
        self.num_states = num_states

        mapping = dict(card_seg=num_segs, card_state=num_states)

        aux_dirpath = self.dirpath / SUBDIRNAME_AUX

        self.gmtk_include_filename, self.gmtk_include_filename_is_new = \
            save_template(self.gmtk_include_filename, RES_INC_TMPL, mapping,
                          aux_dirpath, self.clobber)

    def save_structure(self):
        observation_sub = resource_substitute("observation.tmpl") # XXX: make this a variable

        tracknames = self.tracknames
        num_tracks = self.num_tracks
        num_datapoints = self.num_datapoints
        min_seg_len = self.min_seg_len

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

            # weight scale cannot be more than min_seg_len to avoid
            # artifactual problems
            weight_scale = min(max_num_datapoints_track / num_datapoints_track,
                               min_seg_len)

            # XXX: should avoid a weight line at all when weight_scale == 1.0
            # might avoid some extra multiplication in GMTK
            item = observation_sub(track=track, track_index=track_index,
                                   presence_index=num_tracks+track_index,
                                   weight_scale=weight_scale)
            observation_items.append(item)

        if self.use_dinucleotide:
            dinucleotide_sub = resource_substitute("dinucleotide.tmpl")

            item = dinucleotide_sub(dinucleotide_index=num_tracks*2,
                                    presence_index=num_tracks*2+1)
            observation_items.append(item)

        assert observation_items # must be at least one track
        observations = "\n".join(observation_items)

        mapping = dict(include_filename=self.gmtk_include_filename,
                       observations=observations)

        self.structure_filename, self.structure_filename_is_new = \
            save_template(self.structure_filename, RES_STR_TMPL, mapping,
                          self.dirname, self.clobber)

    def save_observations_chunk(self, float_filepath, int_filepath, float_data,
                                seq_data):
        # input function in GMTK_ObservationMatrix.cc:
        # ObservationMatrix::readBinSentence

        # input per frame is a series of float32s, followed by a series of
        # int32s it is better to optimize both sides here by sticking all
        # the floats in one file, and the ints in another one
        if float_data is None:
            int_data = None
        else:
            mask_missing = isnan(float_data)

            # output -> int_data
            # done in two steps so I can specify output type
            int_data = empty(mask_missing.shape, DTYPE_OBS_INT)
            invert(mask_missing, int_data)

            float_data[mask_missing] = SENTINEL

            float_data.tofile(float_filepath)

        if seq_data is not None:
            extra_int_data = make_dinucleotide_int_data(seq_data)

            # XXXopt: use the correctly sized matrix in the first place
            if int_data is None:
                int_data = extra_int_data
            else:
                int_data = column_stack([int_data, extra_int_data])

        int_data.tofile(int_filepath)

    def subset_metadata_attr(self, name):
        subset_array = getattr(self, name)[self.track_indexes]

        setattr(self, name, subset_array)

    def subset_metadata(self):
        """
        limits all the metadata attributes to only tracks that are used
        """
        subset_metadata_attr = self.subset_metadata_attr
        subset_metadata_attr("mins")
        subset_metadata_attr("maxs")
        subset_metadata_attr("sums")
        subset_metadata_attr("sums_squares")
        subset_metadata_attr("num_datapoints")

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

        chromosome_attrs = chromosome.root._v_attrs

        try:
            mins = chromosome_attrs.mins
            maxs = chromosome_attrs.maxs
            sums = chromosome_attrs.sums
            sums_squares = chromosome_attrs.sums_squares
            num_datapoints = chromosome_attrs.num_datapoints
        except AttributeError:
            # this means there is no data for that chromosome
            return False

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
        # XXX: unsupported until order of things is fixed
        assert not self.split_sequences

        include_coords = self.include_coords

        # observations
        print_obs_filepaths_custom = partial(self.print_obs_filepaths,
                                             float_filelist, int_filelist)
        save_observations_chunk = self.save_observations_chunk
        clobber = self.clobber

        num_tracks = None # this is before any subsetting
        chunk_index = 0
        chunk_coords = []
        num_bases = 0

        use_dinucleotide = "dinucleotide" in self.include_tracknames

        progs_used = self.get_progs_used()

        float_tabwriter = ListWriter(float_tabfile)
        float_tabwriter.writerow(FLOAT_TAB_FIELDNAMES)

        chrom_iterator = iter_chroms_coords(self.h5filenames, include_coords)
        for chrom, filename, chromosome, chr_include_coords in chrom_iterator:
            assert not chromosome.root._v_attrs.dirty

            # if there is no metadata, then skip the chromosome
            if not self.accum_metadata_chromosome(chromosome):
                continue

            track_indexes = self.set_tracknames(chromosome)
            num_tracks = len(track_indexes)

            self.set_num_tracks(num_tracks, use_dinucleotide)

            # observations
            if num_tracks:
                supercontig_walker = walk_continuous_supercontigs(chromosome)
            else:
                supercontig_walker = walk_all_supercontigs(chromosome)

            for supercontig, continuous in supercontig_walker:
                assert continuous is None or continuous.shape[1] >= num_tracks

                supercontig_attrs = supercontig._v_attrs
                supercontig_start = supercontig_attrs.start

                convert_chunks_custom = partial(convert_chunks,
                                                supercontig_attrs)

                if continuous is None:
                    starts = [supercontig_start]
                    ends = [supercontig_attrs.end]
                else:
                    starts = convert_chunks_custom("chunk_starts")
                    ends = convert_chunks_custom("chunk_ends")

                ## iterate through chunks and write
                ## izip so it can be modified in place
                for start_index, start, end in izip(count(), starts, ends):
                    if include_coords:
                        overlaps = find_overlaps(start, end,
                                                 chr_include_coords)
                        len_overlaps = len(overlaps)

                        if len_overlaps == 0:
                            continue
                        elif len_overlaps == 1:
                            start, end = overlaps[0]
                        else:
                            new_starts, new_ends = zip(*overlaps)
                            update_starts(starts, ends, new_starts, new_ends,
                                          start_index)
                            continue # consider the newly split sequences next

                    num_frames = end - start
                    if not MIN_FRAMES <= num_frames <= MAX_FRAMES:
                        text = " skipping segment of length %d" % num_frames
                        print >>sys.stderr, text
                        continue

                    # check that this sequence can fit into all of the
                    # programs that will be used
                    max_mem_usage = max(self.get_mem_usage(num_frames, prog)
                                        for prog in progs_used)
                    if max_mem_usage > MEM_USAGE_LIMIT:
                        if not self.split_sequences:
                            msg = "segment of length %d will take %d memory," \
                                " which is greater than" \
                                " MEM_USAGE_LIMIT" % (num_frames,
                                                      max_mem_usage)
                            raise ValueError(msg)

                        # XXX: I really ought to check that this is
                        # going to always work even for corner cases,
                        # but if it doesn't I am saved by another
                        # split later on

                        # split_sequences was True, so split them
                        num_new_starts = int(ceil(max_mem_usage
                                                  / MEM_USAGE_LIMIT))

                        # // means floor division
                        offset = (num_frames // num_new_starts)
                        new_offsets = arange(num_new_starts) * offset
                        new_starts = start + new_offsets
                        new_ends = append(new_starts[1:], end)

                        update_starts(starts, ends, new_starts, new_ends,
                                      start_index)
                        continue

                    # start: relative to beginning of chromosome
                    # chunk_start: relative to the beginning of
                    # the supercontig
                    chunk_start = start - supercontig_start
                    chunk_end = end - supercontig_start
                    chunk_coords.append((chrom, start, end))

                    float_filepath, int_filepath = \
                        print_obs_filepaths_custom(chrom, chunk_index)

                    row = [float_filepath, str(chunk_index), chrom, str(start),
                           str(end)]
                    float_tabwriter.writerow(row)
                    print >>sys.stderr, " %s (%d, %d)" % (float_filepath,
                                                          start, end)

                    num_bases += end - start

                    # if they don't both exist
                    if not (float_filepath.exists() and int_filepath.exists()):
                        # read rows first into a numpy.array because
                        # you can't do complex imports on a
                        # numpy.Array
                        if continuous is None:
                            cells = None
                        else:
                            min_col = track_indexes.min()
                            max_col = track_indexes.max() + 1
                            col_slice = s_[min_col:max_col]

                            rows = continuous[chunk_start:chunk_end, col_slice]

                            # correct for min_col offset
                            cells = rows[..., track_indexes - min_col]

                        if use_dinucleotide:
                            seq = supercontig.seq
                            len_seq = len(seq)

                            if chunk_end < len_seq:
                                seq_cells = seq[chunk_start:chunk_end+1]
                            elif chunk_end == len(seq):
                                seq_chunk = seq[chunk_start:chunk_end]
                                seq_cells = append(seq_chunk, ord("N"))
                            else:
                                raise ValueError("sequence too short for"
                                                 " supercontig")
                        else:
                            seq_cells = None

                        save_observations_chunk(float_filepath, int_filepath,
                                                cells, seq_cells)

                    chunk_index += 1

        self.subset_metadata()

        self.num_chunks = chunk_index # already has +1 added to it
        self.num_bases = num_bases
        self.chunk_coords = chunk_coords

    def set_num_tracks(self, num_tracks, use_dinucleotide):
        self.num_tracks = num_tracks

        self.use_dinucleotide = use_dinucleotide
        if use_dinucleotide:
            self.num_int_cols = num_tracks + NUM_SEQ_COLS
        else:
            self.num_int_cols = num_tracks

    def open_writable_or_dummy(self, filepath):
        if not filepath or (not self.clobber and filepath.exists()):
            return closing(StringIO()) # dummy output
        else:
            return open(filepath, "w")

    def save_observations(self):
        open_writable_or_dummy = self.open_writable_or_dummy

        with open_writable_or_dummy(self.float_filelistpath) as float_filelist:
            with open_writable_or_dummy(self.int_filelistpath) as int_filelist:
                with open_writable_or_dummy(self.float_tabfilepath) as \
                        float_tabfile:
                    self.write_observations(float_filelist, int_filelist,
                                            float_tabfile)

    def calc_means_vars(self):
        num_datapoints = self.num_datapoints
        means = self.sums / num_datapoints

        # this is an unstable way of calculating the variance,
        # but it should be good enough
        # Numerical Recipes in C, Eqn 14.1.7
        # XXX: best would be to switch to the pairwise parallel method
        # (see Wikipedia)

        self.means = means
        self.vars = (self.sums_squares / num_datapoints) - square(means)

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
        num_states = self.num_states

        return zeros((num_states, num_states))

    def make_linked_cpt(self):
        """
        has all of the seg0_0 -> seg0_1 links, etc.

        also has a seg 0_f -> seg1_0 link which must be overwritten,
        such as by add_cpt()
        """
        return diagflat(ones(self.num_states-1), 1)

    def get_first_state_indexes(self):
        return arange(0, self.num_states, self.min_seg_len)

    def add_final_probs_to_cpt(self, cpt):
        """
        modifies original
        """
        num_states = self.num_states
        num_segs = self.num_segs
        min_seg_len = self.min_seg_len

        prob_self_self = prob_transition_from_expected_len(LEN_SEG_EXPECTED)
        prob_self_other = (1.0 - prob_self_self) / (num_segs - 1)

        final_state_indexes = arange(min_seg_len - 1, num_states, min_seg_len)
        first_state_indexes = self.get_first_state_indexes()

        self_self_coords = [final_state_indexes, final_state_indexes]

        # XXX: this is not extensible to more than 2 segs
        assert num_segs == 2
        self_other_coords = [final_state_indexes, first_state_indexes[::-1]]

        cpt[self_self_coords] = prob_self_self
        cpt[self_other_coords] = prob_self_other

        return cpt

    def make_dirichlet_table(self):
        num_segs = self.num_segs

        probs = self.add_final_probs_to_cpt(self.make_empty_cpt())

        # XXX: the ratio is not exact as num_bases is not the same as
        # the number of base-base transitions. It is surely close
        # enough, though
        total_pseudocounts = self.len_seg_strength * self.num_bases
        pseudocounts_per_row = total_pseudocounts / num_segs

        # astype(int) means flooring the floats
        pseudocounts = (probs * pseudocounts_per_row).astype(int)

        return pseudocounts

    def make_dirichlet_spec(self):
        dirichlet_table = self.make_dirichlet_table()
        items = [make_table_spec(DIRICHLET_FRAG, dirichlet_table)]

        return make_spec("DIRICHLET_TAB", items)

    def make_dense_cpt_start_state_spec(self):
        cpt = zeros((1, self.num_states))

        cpt[0, self.get_first_state_indexes()] = 1.0 / self.num_segs

        return make_table_spec(DENSE_CPT_START_STATE_FRAG, cpt)

    def make_dense_cpt_state_state_spec(self):
        num_segs = self.num_segs

        if self.len_seg_strength > 0:
            frag = "\n".join([DENSE_CPT_STATE_STATE_FRAG, DIRICHLET_STATE_STATE_FRAG])
        else:
            frag = DENSE_CPT_STATE_STATE_FRAG

        cpt = self.add_final_probs_to_cpt(self.make_linked_cpt())
        return make_table_spec(frag, cpt)

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
        # XXX: need to make this work for a bigger min_seg_len, appropriately tied
        table = [self.make_dinucleotide_table_row()
                 for seg_index in xrange(self.num_segs)]

        return make_table_spec(DENSE_CPT_SEG_DINUCLEOTIDE_FRAG, table)

    def make_dense_cpt_spec(self):
        num_segs = self.num_segs

        items = [self.make_dense_cpt_start_state_spec(),
                 self.make_dense_cpt_state_state_spec()]

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

    def save_input_master(self, start_index=None, new=False):
        tracknames = self.tracknames
        num_segs = self.num_segs

        include_filename = self.gmtk_include_filename

        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        if self.len_seg_strength > 0:
            dirichlet_spec = self.make_dirichlet_spec()
        else:
            dirichlet_spec = ""

        dense_cpt_spec = self.make_dense_cpt_spec()

        self.calc_means_vars()

        distribution = self.distribution
        if distribution == DISTRIBUTION_NORM:
            mean_spec = self.make_mean_spec()
            covar_spec = self.make_covar_spec(COVAR_TIED)
            gamma_spec = ""
        elif distribution == DISTRIBUTION_GAMMA:
            mean_spec = ""
            covar_spec = ""

            # XXX: another option is to calculate an ML estimate for
            # the gamma distribution rather than the ML estimate for the
            # mean and converting
            gamma_spec = self.make_gamma_spec()
        else:
            raise ValueError("distribution %s not supported" % distribution)

        mc_spec = self.make_mc_spec()
        mx_spec = self.make_mx_spec()
        name_collection_spec = make_name_collection_spec(num_segs, tracknames)

        params_dirpath = self.dirpath / SUBDIRNAME_PARAMS

        self.input_master_filename, self.input_master_filename_is_new = \
            save_template(input_master_filename, RES_INPUT_MASTER_TMPL,
                          locals(), params_dirpath, self.clobber,
                          start_index)

    def save_dont_train(self):
        self.dont_train_filename = self.save_resource(RES_DONT_TRAIN,
                                                      SUBDIRNAME_AUX)

    def save_output_master(self):
        self.output_master_filename = self.save_resource(RES_OUTPUT_MASTER,
                                                         SUBDIRNAME_PARAMS)

    def save_viterbi_filelist(self):
        dirpath = self.dirpath / SUBDIRNAME_VITERBI
        num_chunks = self.num_chunks

        viterbi_filename_fmt = (PREFIX_VITERBI + make_prefix_fmt(num_chunks)
                               + EXT_OUT)
        viterbi_filenames = [dirpath / viterbi_filename_fmt % index
                            for index in xrange(num_chunks)]

        viterbi_filelistname = dirpath / extjoin("output", EXT_LIST)
        self.viterbi_filelistname = viterbi_filelistname

        with open(viterbi_filelistname, "w") as viterbi_filelist:
            for viterbi_filename in viterbi_filenames:
                print >>viterbi_filelist, viterbi_filename

        self.viterbi_filenames = viterbi_filenames

    def make_posterior_filenames(self):
        make_posterior_filename = self.make_posterior_filename
        chunk_range = xrange(self.num_chunks)

        self.posterior_filenames = map(make_posterior_filename, chunk_range)

    def save_dumpnames(self):
        self.dumpnames_filename = self.save_resource(RES_DUMPNAMES,
                                                     SUBDIRNAME_AUX)

    def save_params(self):
        self.load_include_coords()

        self.make_subdirs(SUBDIRNAMES_EITHER)
        self.make_obs_dir()

        # do first, because it sets self.num_tracks and self.tracknames
        self.save_observations()

        # sets self.chunk_lens, needed for save_structure()
        self.make_chunk_lens()

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

            self.save_dont_train()
            self.save_output_master()

            # might turn off self.train, if the params already exist
            self.set_log_likelihood_filename()

        if identify or posterior:
            self.make_subdirs(SUBDIRNAMES_IDENTIFY)

            self.save_viterbi_filelist()
            self.save_dumpnames()

    def move_results(self, name, src_filename, dst_filename):
        if dst_filename:
            # XXX: i think this should become copy
            move(src_filename, dst_filename)
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

    def make_wig_header_posterior(self, state_name):
        attrs = WIG_ATTRS_POSTERIOR.copy()
        attrs["name"] = WIG_NAME_POSTERIOR % state_name

        return self.make_wig_desc_attrs(attrs,
                                        WIG_DESC_POSTERIOR % state_name)

    def gmtk_out2bed(self):
        bed_filename = self.bed_filename

        if bed_filename is None:
            bed_filename = self.dirpath / BED_FILEBASENAME

        # chunk_coord = (chrom, chromStart, chromEnd)
        zipper = izip(self.viterbi_filenames, self.chunk_coords)
        with gzip_open(bed_filename, "w") as bed_file:
            # XXX: add in browser track line (see SVN revisions
            # previous to 195)
            print >>bed_file, self.make_wig_header_viterbi()

            for gmtk_outfilename, chunk_coord in zipper:
                load_gmtk_out_write_bed(chunk_coord, gmtk_outfilename,
                                        bed_file)

    def posterior2wig(self):
        infilenames = self.posterior_filenames

        range_num_segs = xrange(self.num_segs)
        wig_filenames = map(self.make_posterior_wig_filename, range_num_segs)

        # XXX: repetitive with gmtk_out2bed
        zipper = izip(infilenames, self.chunk_coords)

        wig_files_unentered = [gzip_open(wig_filename, "w")
                               for wig_filename in wig_filenames]

        with nested(*wig_files_unentered) as wig_files:
            for state_index, wig_file in enumerate(wig_files):
                print >>wig_file, self.make_wig_header_posterior(state_index)

            for infilename, chunk_coord in zipper:
                load_posterior_write_wig(chunk_coord, infilename, wig_files)

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

    def make_acc_filename(self, start_index, chunk_index):
        return self.make_filename(PREFIX_ACC, start_index, chunk_index,
                                  EXT_BIN, subdirname=SUBDIRNAME_ACC)

    def make_posterior_filename(self, chunk_index):
        return self.make_filename(PREFIX_POSTERIOR, chunk_index, EXT_TXT,
                                  subdirname=SUBDIRNAME_POSTERIOR)

    def make_job_name_train(self, start_index, round_index, chunk_index):
        return "%s%d.%d.%s.%s.%s" % (PREFIX_JOB_NAME_TRAIN, start_index,
                                     round_index, chunk_index,
                                     self.dirpath.name, UUID)

    def make_job_name_identify(self, prefix, chunk_index):
        return "%s%d.%s.%s" % (prefix, chunk_index, self.dirpath.name,
                               UUID)

    def make_gmtk_kwargs(self):
        res = dict(strFile=self.structure_filename,
                   verbosity=self.verbosity,
                   jtFile=self.jt_info_filename)

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

    def parse_res_usage(self):
        # dict
        # key: (program, num_tracks)
        # val: dict
        #      key: (island_base, island_lst)
        #      val: (mem_per_obs, cpu_per_obs)
        res_usage = defaultdict(dict)

        with open(self.res_usage_filename) as infile:
            reader = DictReader(infile)
            for row in reader:
                # XXX: need to add num_segs
                program = row["program"]
                num_tracks = int(row["num_tracks"])
                island_base = int(row["island_base"])
                island_lst = int(row["island_lst"])
                mem_per_obs = int(row["mem_per_obs"])
                cpu_per_obs = float(row["mem_per_obs"])

                # XXX: need to add num_segs
                top_specifier = (program, num_tracks)
                island_specifier = (island_base, island_lst)
                data = (mem_per_obs, cpu_per_obs)

                res_usage[top_specifier][island_specifier] = data

        self.res_usage = res_usage

    def get_mem_per_obs(self, prog, num_tracks):
        program = prog.prog

        island_specifier = (ISLAND_BASE, ISLAND_LST)

        # will fail if the tuple is not a key three
        res_usage = self.res_usage[program, num_tracks][island_specifier]
        return res_usage[0]

    def get_mem_usage(self, chunk_len, prog=EM_TRAIN_PROG):
        """
        returns an int
        """
        mem_per_obs = self.mem_per_obs[prog.prog]

        # assume maximum memory usage until returned results say otherwise
        if mem_per_obs is None:
            return MEM_USAGE_LIMIT

        res = chunk_len * mem_per_obs
        if res > MEM_USAGE_LIMIT:
            msg = "segment of length %d will take %d memory," \
                " which is greater than MEM_USAGE_LIMIT" % (chunk_len, res)

            raise ValueError(msg)

        return res

    def make_chunk_lens(self):
        self.chunk_lens = [end - start
                           for chr, start, end in self.chunk_coords]

    def chunk_lens_sorted(self, reverse=False):
        """
        yields (chunk_index, chunk_mem_usage)

        if reverse: sort chunks by decreasing size, so the most
        difficult chunks are dropped in the queue first
        """
        chunk_lens = self.chunk_lens
        zipper = sorted(izip(chunk_lens, count()), reverse=reverse)

        # XXX: use itertools instead of a generator
        for chunk_len, chunk_index in zipper:
            yield chunk_index, chunk_len

    def queue_gmtk(self, prog, kwargs, job_name, mem_usage, native_specs={},
                   output_filename=None):
        if mem_usage > MEM_USAGE_LIMIT:
            msg = "%s with a request of %d memory, which is greater" \
                " than MEM_USAGE_LIMIT" % (prog.prog, mem_usage)

            if self.skip_large_mem_usage:
                print >>sys.stderr, "skipping " + msg
                return

            raise ValueError("won't queue " + msg)

        gmtk_cmdline = prog.build_cmdline(options=kwargs)

        # convoluted so I don't have to deal with a lot of escaping issues
        cmdline = BASH_CMDLINE + ['%s "$@"' % gmtk_cmdline[0]] + gmtk_cmdline

        print >>self.cmdline_file, " ".join(gmtk_cmdline)

        if self.dry_run:
            return None

        session = self.session
        job_tmpl = session.createJobTemplate()

        # shouldn't this be jobName? not in the Python DRMAA implementation
        # XXX: report upstream
        job_tmpl.name = job_name

        job_tmpl.remoteCommand = ENV_CMD
        job_tmpl.args = cmdline

        if output_filename is None:
            output_filename = self.output_dirpath / job_name
        job_tmpl.outputPath = ":" + output_filename
        job_tmpl.errorPath = ":" + (self.error_dirpath / job_name)

        set_cwd_job_tmpl(job_tmpl)

        res_req = make_res_req(make_mem_req(mem_usage))
        job_tmpl.nativeSpecification = make_native_spec(l=res_req,
                                                        **native_specs)

        return session.runJob(job_tmpl)

    def queue_train(self, start_index, round_index,
                    chunk_index, mem_usage, hold_jid=None, **kwargs):
        """
        this calls Runner.queue_gmtk() and returns None if the
        mem_usage is too large, and self.skip_large_mem_usage is True.
        If it is False, a ValueError is raised.
        """
        kwargs["inputMasterFile"] = self.input_master_filename

        prog = self.train_prog
        name = self.make_job_name_train(start_index, round_index, chunk_index)
        native_specs = dict(hold_jid=hold_jid)

        return self.queue_gmtk(prog, kwargs, name, mem_usage, native_specs)

    def queue_train_parallel(self, input_params_filename, start_index,
                             round_index, **kwargs):
        queue_train_custom = partial(self.queue_train, start_index,
                                     round_index)

        kwargs["cppCommandOptions"] = make_cpp_options(input_params_filename)

        res = [] # task ids

        prog = EM_TRAIN_PROG.prog

        # first round is done smallest chunk first
        reverse = self.mem_per_obs[prog] is not None

        for chunk_index, chunk_len in self.chunk_lens_sorted(reverse):
            acc_filename = self.make_acc_filename(start_index, chunk_index)
            kwargs_chunk = dict(trrng=chunk_index, storeAccFile=acc_filename,
                                **kwargs)

            # -dirichletPriors T only on the first chunk
            kwargs_chunk["dirichletPriors"] = (chunk_index == 0)

            # reread it here so that it can be reset in the middle of
            # the stream
            mem_usage = self.get_mem_usage(chunk_len)
            jobid = queue_train_custom(chunk_index, mem_usage, **kwargs_chunk)

            assert jobid is not None

            if self.mem_per_obs[prog] is None:
                self.wait_job(jobid)

                self.set_mem_per_obs(jobid, chunk_len, prog)

                # don't need to wait for this jobid again, so it's not
                # appended to res
            else:
                res.append(jobid)

        return res

    def queue_train_bundle(self, parallel_jobids,
                           input_params_filename, output_params_filename,
                           start_index, round_index, **kwargs):
        """bundle step: take parallel accumulators and combine them
        """
        acc_filename = self.make_acc_filename(start_index,
                                              GMTK_INDEX_PLACEHOLDER)

        cpp_options = make_cpp_options(input_params_filename,
                                       output_params_filename)

        kwargs = \
            dict(outputMasterFile=self.output_master_filename,
                 cppCommandOptions=cpp_options,
                 trrng="nil",
                 loadAccRange="0:%s" % (self.num_chunks-1),
                 loadAccFile=acc_filename,
                 **kwargs)

        if self.dry_run:
            hold_jid = None
        else:
            hold_jid = ",".join(parallel_jobids)

        return self.queue_train(start_index, round_index,
                                NAME_BUNDLE_PLACEHOLDER, MEM_USAGE_BUNDLE,
                                hold_jid, **kwargs)

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

    def run_triangulate(self):
        prog = self.prog_factory(TRIANGULATE_PROG)

        structure_filename = self.structure_filename
        triangulation_filename = self.triangulation_filename
        if not triangulation_filename:
            triangulation_filename = extjoin(structure_filename, EXT_TRIFILE)
            self.triangulation_filename = triangulation_filename

        kwargs = dict(strFile=structure_filename,
                      outputTriangulatedFile=triangulation_filename,
                      verbosity=self.verbosity)

        # XXX: need exist/clobber logic here
        # XXX: repetitive with queue_gmtk
        cmdline = prog.build_cmdline(options=kwargs)
        print >>self.cmdline_file, " ".join(cmdline)

        prog(**kwargs)

        if not self.posterior_triangulation_filename and not self.dry_run:
            self.save_posterior_triangulation()

    def run_train_round(self, start_index, round_index, **kwargs):
        """
        returns None: normal
        returns not None: abort
        """
        last_params_filename = self.last_params_filename
        curr_params_filename = extjoin(self.params_filename, str(round_index))

        parallel_jobids = \
            self.queue_train_parallel(last_params_filename, start_index,
                                      round_index, **kwargs)

        # will be None if mem_usage > MEM_USAGE_LIMIT and
        # self.skip_large_mem_usage
        bundle_jobid = \
            self.queue_train_bundle(parallel_jobids, last_params_filename,
                                    curr_params_filename, start_index,
                                    round_index,
                                    llStoreFile=self.log_likelihood_filename,
                                    **kwargs)

        self.last_params_filename = curr_params_filename

        if self.dry_run:
            return False

        return self.wait_job(bundle_jobid, parallel_jobids)

    def run_train_start(self):
        # make new files if you have more than one random start
        new = self.make_new_params

        start_index = self.start_index

        self.save_input_master(start_index, new)
        self.set_params_filename(start_index, new)
        self.set_log_likelihood_filename(start_index, new)
        self.set_output_dirpaths(start_index)

        last_log_likelihood = NINF
        log_likelihood = NINF
        round_index = 0

        self.last_params_filename = None

        kwargs = self.train_kwargs

        while (round_index < self.max_em_iters and
               is_training_progressing(last_log_likelihood, log_likelihood)):
            round_res = self.run_train_round(start_index, round_index,
                                             **kwargs)
            if round_res is not None:
                return (None, None, None, None)

            last_log_likelihood = log_likelihood
            log_likelihood = self.load_log_likelihood()

            print >>sys.stderr, "log likelihood = %s" % log_likelihood

            round_index += 1

        # log_likelihood and a list of src_filenames to save
        return (log_likelihood, self.input_master_filename,
                self.last_params_filename, self.log_likelihood_filename)

    def wait_job(self, jobid, kill_jobids=[]):
        """
        wait for bundle to finish
        """
        # XXXopt: polling in each thread is a bad way to do this
        # it would be best to use session.synchronize() centrally
        # and communicate to each thread when its job is done

        # the very best thing would be to eliminate the GIL lock
        # in the DRMAA wrapper
        job_info = None
        session = self.session
        interrupt_event = self.interrupt_event

        control = session.control
        terminate = session.TERMINATE
        while not job_info:
            try:
                job_info = session.wait(jobid, session.TIMEOUT_NO_WAIT)
            except ExitTimeoutError:
                # ExitTimeoutError: not ready yet
                interrupt_event.wait(SESSION_WAIT_TIMEOUT)
            except ValueError:
                # ValueError: the job terminated abnormally
                # so interrupt everybody
                if self.keep_going:
                    return False
                else:
                    interrupt_event.set()
                    raise

            if interrupt_event.isSet():
                for jobid in kill_jobids + [jobid]:
                    try:
                        print >>sys.stderr, "killing job %s" % jobid
                        control(jobid, terminate)
                    except BaseException, err:
                        print >>sys.stderr, ("ignoring exception: %r"
                                             % err)
                raise KeyboardInterrupt

    def run_train(self):
        self.train_prog = self.prog_factory(EM_TRAIN_PROG)

        self.train_kwargs = dict(objsNotToTrain=self.dont_train_filename,
                                 maxEmIters=1,
                                 island=ISLAND,
                                 lldp=LOG_LIKELIHOOD_DIFF_FRAC*100.0,
                                 triFile=self.triangulation_filename,
                                 **self.make_gmtk_kwargs())
        if ISLAND:
            self.train_kwargs["base"] = ISLAND_BASE
            self.train_kwargs["lst"] = ISLAND_LST

        # save the destination file for input_master as we will be
        # generating new input masters for each start
        #
        # len(dst_filenames) == len(TRAIN_ATTRNAMES) == len(return value
        # of Runner.run_train_start())-1. This is asserted below.

        random_starts = self.random_starts
        assert random_starts >= 1

        # XXX: why did I have "if random_starts == 1:" preceding this line?
        self.save_input_master()

        # should I make new parameters in each thread?
        self.make_new_params = (self.random_starts > 1
                                or isinstance(self.num_segs, slice))

        if self.input_master_filename_is_new:
            input_master_filename = self.input_master_filename
        else:
            input_master_filename = None

        dst_filenames = [input_master_filename,
                         self.params_filename,
                         self.log_likelihood_filename]

        interrupt_event = Event()

        num_segs_range = slice2range(self.num_segs)

        # XXX: Python 2.6 use itertools.product()
        enumerator = enumerate((num_seg, seg_start_index)
                               for num_seg in num_segs_range
                               for seg_start_index in xrange(random_starts))

        threads = []
        with Session() as session:
            try:
                for start_index, (num_seg, seg_start_index) in enumerator:
                    thread = RandomStartThread(self, session, start_index,
                                               num_seg, interrupt_event)
                    thread.start()
                    threads.append(thread)

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
                interrupt_event.set()
                for thread in threads:
                    thread.join()

                raise

        self.proc_train_results(start_params, dst_filenames)

    def proc_train_results(self, start_params, dst_filenames):
        if self.dry_run:
            return

        # finds the max by log_likelihood
        src_filenames = max(start_params)[1:]

        if None in src_filenames:
            raise ValueError, "all training threads failed"

        assert LEN_TRAIN_ATTRNAMES == len(src_filenames) == len(dst_filenames)

        zipper = zip(TRAIN_ATTRNAMES, src_filenames, dst_filenames)
        for name, src_filename, dst_filename in zipper:
            self.move_results(name, src_filename, dst_filename)

    def _queue_identify(self, jobids, chunk_index, kwargs_chunk,
                        chunk_mem_usage, prefix_job_name, prog, kwargs_func,
                        output_filename=None):
        job_name = self.make_job_name_identify(prefix_job_name, chunk_index)

        kwargs = kwargs_chunk.copy()
        kwargs.update(kwargs_func)

        jobid = self.queue_gmtk(prog, kwargs, job_name, chunk_mem_usage,
                                output_filename=output_filename)
        if jobid is not None:
            jobids.append(jobid)

        return jobid

    def run_identify_posterior(self):
        if not self.input_master_filename:
            self.save_input_master()

        params_filename = self.params_filename

        prog_viterbi = self.prog_factory(VITERBI_PROG)
        prog_posterior = self.prog_factory(POSTERIOR_PROG)

        identify_kwargs = \
            dict(inputMasterFile=self.input_master_filename,

                 cppCommandOptions=make_cpp_options(params_filename),
                 **self.make_gmtk_kwargs())

        self.set_output_dirpaths("identify")

        # XXXopt: might be inefficient to write to standard output and
        # save via redirection
        viterbi_kwargs = dict(triFile=self.triangulation_filename,
                              vitValsFile="-") # standard output, saved by SGE

        self.make_posterior_filenames()

        viterbi_filenames = self.viterbi_filenames
        posterior_filenames = self.posterior_filenames

        get_mem_usage = self.get_mem_usage

        chunk_lens = self.chunk_lens

        # XXX: kill submitted jobs on exception
        jobids = []
        with Session() as session:
            self.session = session

            posterior_kwargs = \
                dict(triFile=self.posterior_triangulation_filename,
                     doDistributeEvidence=True,
                     **self.get_posterior_clique_print_ranges())

            for chunk_index, chunk_len in self.chunk_lens_sorted():
                identify_kwargs_chunk = dict(dcdrng=chunk_index,
                                             **identify_kwargs)
                _queue_identify = partial(self._queue_identify, jobids,
                                          chunk_index, identify_kwargs_chunk)

                # XXX: next two segments are highly duplicative!
                if self.identify:
                    mem_usage = get_mem_usage(chunk_len, VITERBI_PROG)
                    viterbi_filename = viterbi_filenames[chunk_index]
                    jobid = _queue_identify(mem_usage, PREFIX_JOB_NAME_VITERBI,
                                            prog_viterbi, viterbi_kwargs,
                                            viterbi_filename)
                    prog = VITERBI_PROG.prog
                    if self.mem_per_obs[prog] is None:
                        self.wait_job(jobid)
                        self.set_mem_per_obs(jobid, chunk_len, prog)

                if self.posterior:
                    mem_usage = get_mem_usage(chunk_len, POSTERIOR_PROG)
                    posterior_filename = posterior_filenames[chunk_index]
                    jobid = _queue_identify(mem_usage,
                                            PREFIX_JOB_NAME_POSTERIOR,
                                            prog_posterior, posterior_kwargs,
                                            posterior_filename)
                    prog = POSTERIOR_PROG.prog
                    if self.mem_per_obs[prog] is None:
                        self.wait_job(jobid)
                        self.set_mem_per_obs(jobid, chunk_len, prog)


            # XXX: ask on DRMAA mailing list--how to allow
            # KeyboardInterrupt here?

            if self.dry_run:
                return

            session.synchronize(jobids, session.TIMEOUT_WAIT_FOREVER, True)

        # XXXopt: parallelize
        if self.identify:
            self.gmtk_out2bed()

        if self.posterior:
            self.posterior2wig()

    def run(self):
        """
        main run, after dirname is specified

        this is exposed so that it can be overriden in a subclass
        """
        # XXXopt: use binary I/O to gmtk rather than ascii

        self.dirpath = path(self.dirname)
        self.parse_res_usage()
        self.save_params()

        self.make_subdir(SUBDIRNAME_LOG)
        cmdline_filename = self.make_filename(PREFIX_CMDLINE, EXT_SH,
                                              subdirname=SUBDIRNAME_LOG)

        with open(cmdline_filename, "w") as cmdline_file:
            # XXX: works around a pyflakes bug; report
            self.cmdline_file = cmdline_file
            if self.triangulate:
                self.run_triangulate()

            if self.train:
                self.run_train()

            if self.identify or self.posterior:
                self.run_identify_posterior()

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

    usage = "%prog [OPTION]... H5FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    with OptionGroup(parser, "Data subset") as group:
        group.add_option("-t", "--track", action="append", default=[],
                         metavar="TRACK",
                         help="append TRACK to list of tracks to use"
                         " (default all)")

        # This is a 0-based file.
        # I know because ENm008 starts at position 0 in encodeRegions.txt.gz
        group.add_option("--include-coords", metavar="FILE",
                         help="limit to genomic coordinates in FILE")

    with OptionGroup(parser, "Model files") as group:
        group.add_option("-i", "--input-master", metavar="FILE",
                         help="use or create input master in FILE")

        group.add_option("-s", "--structure", metavar="FILE",
                         help="use or create structure in FILE")

        group.add_option("-p", "--trainable-params", metavar="FILE",
                         help="use or create trainable parameters in FILE")

        group.add_option("--resource-profile", metavar="FILE",
                         help="load segway-res-usage profile from FILE")

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

        group.add_option("-N", "--num-segs", type=slice,
                         default=NUM_SEGS, metavar="SLICE",
                         help="make SLICE segment classes"
                         " (default %d)" % NUM_SEGS)

        group.add_option("--min-seg-length", type=int,
                         default=MIN_SEG_LEN, metavar="NUM",
                         help="enforce minimum segment length of NUM"
                         " (default %d)" % MIN_SEG_LEN)

        group.add_option("--prior-strength", type=float,
                         default=PRIOR_STRENGTH, metavar="RATIO",
                         help="use RATIO times the number of data counts as"
                         " the number of pseudocounts for the segment length"
                         " prior (default %d)" % PRIOR_STRENGTH)

        group.add_option("-v", "--verbosity", type=int, default=VERBOSITY,
                         metavar="NUM",
                         help="show messages with verbosity NUM")

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
        group.add_option("-S", "--split-sequences", action="store_true",
                         help="split up sequences that are too large to fit" \
                         " into memory")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    runner = Runner()

    runner.h5filenames = args
    runner.dirname = options.directory
    runner.obs_dirname = options.observations
    runner.bed_filename = options.bed

    runner.input_master_filename = options.input_master
    runner.structure_filename = options.structure
    runner.params_filename = options.trainable_params
    runner.include_coords_filename = options.include_coords

    if options.resource_profile:
        runner.res_usage_filename = options.resource_profile

    runner.distribution = options.distribution
    runner.random_starts = options.random_starts
    runner.len_seg_strength = options.prior_strength
    runner.include_tracknames = options.track
    runner.verbosity = options.verbosity
    runner.min_seg_len = options.min_seg_length
    runner.num_segs = options.num_segs

    runner.clobber = options.clobber
    runner.train = not options.no_train
    runner.identify = not options.no_identify
    runner.posterior = not options.no_posterior
    runner.dry_run = options.dry_run
    runner.keep_going = options.keep_going
    runner.split_sequences = options.split_sequences

    return runner()

if __name__ == "__main__":
    sys.exit(main())
