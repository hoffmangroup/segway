#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from cStringIO import StringIO
from collections import defaultdict
from contextlib import closing, contextmanager
from copy import copy
from errno import EEXIST, ENOENT
from functools import partial
from itertools import count, izip
from math import ceil, floor, log10
from os import extsep
from random import random
from shutil import move
from string import Template
from struct import calcsize, unpack
import sys
from threading import Thread

from DRMAA import Session as _Session
from numpy import amin, amax, array, isnan, NINF
from numpy.random import uniform
from optbuild import (Mixin_NoConvertUnderscore,
                      OptionBuilder_ShortOptWithSpace,
                      OptionBuilder_ShortOptWithSpace_TF)
from path import path
from tables import openFile

from ._util import (data_filename, data_string, get_tracknames, gzip_open,
                    init_num_obs, NamedTemporaryDir, PKG,
                    walk_continuous_supercontigs)

# XXX: should be options
NUM_SEGS = 2
MAX_EM_ITERS = 100
VERBOSITY = 0 # XXX: should vary based on DRMAA submission or not
TEMPDIR_PREFIX = PKG + "-"
COVAR_TIED = True # would need to expand to MC, MX to change
MAX_CHUNKS = 1000
ISLAND = False
WIG_DIRNAME = "out"

LOG_LIKELIHOOD_DIFF_FRAC = 1e-5

# for extra memory savings, set to (False) or (not ISLAND)
COMPONENT_CACHE = True

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2
MAX_FRAMES = 1000000000 # 1 billion
MEM_REQ_PARALLEL = "10.5G"
MEM_REQ_BUNDLE = "500M"
RES_REQ_FMT = "mem_requested=%s"

# for a four-way model
MEM_REQ_INTERCEPT_ISLAND = 17255542
MEM_REQ_SLOPE_ISLAND = 4782

MEM_REQ_INTERCEPT = 14442884
MEM_REQ_SLOPE = 5768

# defaults
RANDOM_STARTS = 1

# replace NAN with SENTINEL to avoid warnings
# I chose this value because it is small, and len(str(SENTINEL)) is short
SENTINEL = -5.42e34

ACC_FILENAME_FMT = "acc.%s.%s.bin"
GMTK_INDEX_PLACEHOLDER = "@D"

# programs
ENV_CMD = "/usr/bin/env"
BASH_CMD = "bash"
EM_TRAIN_CMD = "gmtkEMtrainNew"
EM_TRAIN_CMD_STR = '%s "$@"' % EM_TRAIN_CMD

BASH_CMDLINE = [BASH_CMD, "--login", "-c"]
EM_TRAIN_CMDLINE = BASH_CMDLINE + [EM_TRAIN_CMD_STR]

TRIANGULATE_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_ShortOptWithSpace_TF(EM_TRAIN_CMD)
VITERBI_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkViterbiNew")
NATIVE_SPEC_PROG = (Mixin_NoConvertUnderscore
                    + OptionBuilder_ShortOptWithSpace)() # do not run

NATIVE_SPEC_DEFAULT = dict(q="all.q")

OPT_USE_TRAINABLE_PARAMS = "-DUSE_TRAINABLE_PARAMS"

# extensions and suffixes
EXT_GZ = "gz"
EXT_LIKELIHOOD = "ll"
EXT_LIST = "list"
EXT_OBS = "obs"
EXT_OUT = "out"
EXT_PARAMS = "params"
EXT_WIG = "wig"

PREFIX_LIKELIHOOD = "likelihood"
PREFIX_LIST = "features"
PREFIX_CHUNK = "chunk"
PREFIX_PARAMS = "params"

SUFFIX_GZ = extsep + EXT_GZ
SUFFIX_LIST = extsep + EXT_LIST
SUFFIX_OBS = extsep + EXT_OBS
SUFFIX_OUT = extsep + EXT_OUT

FEATURE_FILELISTBASENAME = extsep.join([PREFIX_LIST, EXT_LIST])

# templates and formats
RES_STR_TMPL = "seg.str.tmpl"
RES_INPUT_MASTER_TMPL = "input.master.tmpl"
RES_DONT_TRAIN = "dont_train.list"
RES_INC = "seg.inc"

DENSE_CPT_START_SEG_FRAG = "0 start_seg 0 CARD_SEG"
DENSE_CPT_SEG_SEG_FRAG = "1 seg_seg 1 CARD_SEG CARD_SEG"

MEAN_TMPL = "$index mean_${seg}_${obs} 1 ${rand}"

COVAR_TMPL_TIED = "$index covar_${obs} 1 ${rand}"
COVAR_TMPL_UNTIED = "$index covar_${seg}_${obs} 1 ${rand}" # unused as of yet

MC_TMPL = "$index 1 COMPONENT_TYPE_DIAG_GAUSSIAN" \
    " mc_${seg}_${obs} mean_${seg}_${obs} covar_${obs}"
MX_TMPL = "$index 1 mx_${seg}_${obs} 1 dpmf_always mc_${seg}_${obs}"

NAME_COLLECTION_TMPL = "$obs_index collection_seg_${obs} 2"
NAME_COLLECTION_CONTENTS_TMPL = "mx_${seg}_${obs}"

TRACK_FMT = "browser position %s:%s-%s"
FIXEDSTEP_FMT = "fixedStep chrom=%s start=%s step=1 span=1"

# XXX: this could be specified as a dict instead
WIG_HEADER = 'track type=wiggle_0 name=%s ' \
    'description="%s segmentation of %%s" visibility=dense viewLimits=0:1 ' \
    'autoScale=off' % (PKG, PKG)

TRAIN_ATTRNAMES = ["input_master_filename", "params_filename"]

def extjoin(*args):
    return extsep.join(args)

def extjoin_not_none(*args):
    return extjoin(*[str(arg) for arg in args
                     if arg is not None])


def maybe_gzip_open(filename, *args, **kwargs):
    if filename.endswith(SUFFIX_GZ):
        return gzip_open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)

# XXX: suggest upstream as addition to DRMAA-python
@contextmanager
def Session(*args, **kwargs):
    res = _Session()
    res.init(*args, **kwargs)

    try:
        yield res
    finally:
        res.exit()

def convert_chunks(attrs, name):
    supercontig_start = attrs.start
    edges_array = getattr(attrs, name) + supercontig_start
    return edges_array.tolist()

def is_training_progressing(last_ll, curr_ll,
                            min_ll_diff_frac=LOG_LIKELIHOOD_DIFF_FRAC):
    # using x !< y instead of x >= y to give the right default answer
    # in the case of NaNs
    return not abs((curr_ll - last_ll)/last_ll) < min_ll_diff_frac

def save_template(filename, resource, mapping, dirname=None,
                  delete_existing=False, start_index=None):
    """
    creates a temporary file if filename is None or empty
    """
    if filename:
        if not delete_existing and path(filename).exists():
            return filename
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

    return filename

def accum_extrema(chromosome, mins, maxs):
    chromosome_attrs = chromosome.root._v_attrs
    chromosome_mins = chromosome_attrs.mins
    chromosome_maxs = chromosome_attrs.maxs

    if mins is None:
        mins = chromosome_mins
        maxs = chromosome_maxs
    else:
        mins = amin([chromosome_mins, mins], 0)
        maxs = amax([chromosome_maxs, maxs], 0)

    return mins, maxs

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

def name_job(job_tmpl, start_index, round_index, chunk_index):
    # shouldn't this be jobName? not in SGE's DRMAA implementation
    # XXX: report upstream
    job_tmpl.name = "emt%d.%d.%s" % (start_index, round_index, chunk_index)

def make_mem_req(len):
    res = MEM_REQ_SLOPE * len + MEM_REQ_INTERCEPT

    return "%dM" % ceil(res / 2**20)

def make_cpp_options(params_filename):
    if params_filename:
        return OPT_USE_TRAINABLE_PARAMS
    else:
        return None

def make_native_spec(**kwargs):
    options = NATIVE_SPEC_DEFAULT.copy()
    options.update(kwargs)

    return " ".join(NATIVE_SPEC_PROG.build_args(options=options))

def make_spec(name, items):
    items[:0] = ["%s_IN_FILE inline" % name, str(len(items)), ""]

    return "\n".join(items) + "\n"

# def make_dt_spec(num_obs):
#     return make_spec("DT", ["%d seg_obs%d BINARY_DT" % (index, index)
#                             for index in xrange(num_obs)])

def make_items_multiseg(tmpl, num_segs, num_obs, data=None):
    substitute = Template(tmpl).substitute

    res = []

    for seg_index in xrange(num_segs):
        seg = "seg%d" % seg_index
        for obs_index in xrange(num_obs):
            obs = "obs%d" % obs_index
            mapping = dict(seg=seg, obs=obs,
                           seg_index=seg_index, obs_index=obs_index,
                           index=num_obs*seg_index + obs_index)
            if data is not None:
                mapping["rand"] = data[seg_index, obs_index]

            res.append(substitute(mapping))

    return res

def make_spec_multiseg(name, *args, **kwargs):
    return make_spec(name, make_items_multiseg(*args, **kwargs))

# XXX: reimplement in numpy
def make_normalized_random_rows(num_rows, num_cols):
    res = []

    for row_index in xrange(num_rows):
        row_raw = []
        for col_index in xrange(num_cols):
            row_raw.append(random())

        total = sum(row_raw)

        res.extend([item_raw/total for item_raw in row_raw])

    return res

def make_random_spec(frag, *args, **kwargs):
    random_rows = make_normalized_random_rows(*args, **kwargs)

    return " ".join([frag] + [str(row) for row in random_rows])

def make_dense_cpt_start_seg_spec(num_segs):
    return make_random_spec(DENSE_CPT_START_SEG_FRAG, 1, num_segs)

def make_dense_cpt_seg_seg_spec(num_segs):
    return make_random_spec(DENSE_CPT_SEG_SEG_FRAG, num_segs, num_segs)

def make_dense_cpt_spec(num_segs):
    items = [make_dense_cpt_start_seg_spec(num_segs),
             make_dense_cpt_seg_seg_spec(num_segs)]

    return make_spec("DENSE_CPT", items)

def make_rands(low, high, num_segs):
    assert len(low) == len(high)

    # size parameter is so that we always get an array, even if it
    # has shape = (1,)
    return array([uniform(low, high, len(low))
                  for seg_index in xrange(num_segs)])

def make_mean_spec(num_segs, num_obs, mins, maxs):
    rands = make_rands(mins, maxs, num_segs)

    return make_spec_multiseg("MEAN", MEAN_TMPL, num_segs, num_obs, rands)

def make_covar_spec(num_segs, num_obs, mins, maxs, tied):
    if tied:
        num_segs = 1
        tmpl = COVAR_TMPL_TIED
    else:
        tmpl = COVAR_TMPL_UNTIED

    # always start with maximum variance
    data = array([maxs - mins for seg_index in xrange(num_segs)])

    return make_spec_multiseg("COVAR", tmpl, num_segs, num_obs, data)

def make_mc_spec(num_segs, num_obs):
    return make_spec_multiseg("MC", MC_TMPL, num_segs, num_obs)

def make_mx_spec(num_segs, num_obs):
    return make_spec_multiseg("MX", MX_TMPL, num_segs, num_obs)

def make_name_collection_spec(num_segs, num_obs):
    num_segs = NUM_SEGS
    substitute = Template(NAME_COLLECTION_TMPL).substitute
    substitute_contents = Template(NAME_COLLECTION_CONTENTS_TMPL).substitute

    items = []

    for obs_index in xrange(num_obs):
        obs = "obs%d" % obs_index

        mapping = dict(obs=obs, obs_index=obs_index)

        contents = [substitute(mapping)]
        for seg_index in xrange(num_segs):
            seg = "seg%d" % seg_index
            mapping = dict(seg=seg, obs=obs,
                           seg_index=seg_index, obs_index=obs_index)

            contents.append(substitute_contents(mapping))
        items.append("\n".join(contents))

    return make_spec("NAME_COLLECTION", items)

def make_prefix_fmt(num_filenames):
    # make sure there aresufficient leading zeros
    return "%%0%dd." % (int(floor(log10(num_filenames))) + 1)

def read_gmtk_out(infile):
    data = infile.read()

    # @L: uint32/uint64 (@ is the default modifier: native size/alignment)
    # =L: uint32 (standard size/alignment)
    fmt = "=%dL" % (len(data) / calcsize("=L"))
    return unpack(fmt, data)

def write_wig(outfile, output, (chrom, start, end), tracknames):
    # convert from zero- to one-based
    start_1based = start + 1

    print >>outfile, TRACK_FMT % (chrom, start_1based, end)
    print >>outfile, WIG_HEADER % ", ".join(tracknames)
    print >>outfile, FIXEDSTEP_FMT % (chrom, start_1based)

    print >>outfile, "\n".join(map(str, output))

def load_gmtk_out_save_wig(chunk_coord, gmtk_outfilename, wig_filename,
                           tracknames):
    with open(gmtk_outfilename) as gmtk_outfile:
        data = read_gmtk_out(gmtk_outfile)

        with gzip_open(wig_filename, "w") as wig_file:
            return write_wig(wig_file, data, chunk_coord, tracknames)

def set_cwd_job_tmpl(job_tmpl):
    job_tmpl.workingDirectory = path.getcwd()

class RandomStartThread(Thread):
    def __init__(self, runner, start_index):
        raise NotImplementedError

        self.runner = copy(runner)
        self.start_index = start_index

    def run(self):
        self.runner.run_train_start()
        # XXX return likelihood somehow

class Runner(object):
    def __init__(self, **kwargs):
        # filenames
        self.h5filenames = None
        self.feature_filelistpath = None

        self.gmtk_include_filename = None
        self.input_master_filename = None
        self.structure_filename = None

        self.params_filename = None
        self.dirname = None
        self.is_dirname_temp = False
        self.log_likelihood_filename = None
        self.dont_train_filename = None

        self.dumpnames_filename = None
        self.output_filelistname = None
        self.output_filenames = None

        self.obs_dirname = None
        self.wig_dirname = None

        self.include_coords_filename = None

        # data
        # a "chunk" is what GMTK calls a segment
        self.num_chunks = None
        self.chunk_coords = None
        self.mins = None
        self.maxs = None
        self.chunk_mem_reqs = None
        self.tracknames = None

        # variables
        self.num_segs = NUM_SEGS
        self.random_starts = RANDOM_STARTS

        # flags
        self.delete_existing = False
        self.triangulate = True
        self.train = True # EM train # this should become an int for num_starts
        self.identify = True # viterbi
        self.dry_run = False

        # functions
        self.train_prog = None

        self.__dict__.update(kwargs)

    def load_log_likelihood(self):
        with open(self.log_likelihood_filename) as infile:
            return float(infile.read().strip())

    def load_include_coords(self):
        filename = self.include_coords_filename

        if not filename:
            self.include_coords = None
            return

        coords = defaultdict(list)

        with maybe_gzip_open(filename) as infile:
            for line in infile:
                words = line.rstrip().split()
                chrom = words[0]
                start = int(words[1])
                end = int(words[2])

                coords[chrom].append((start, end))

        self.include_coords = dict((chrom, array(coords_list))
                                   for chrom, coords_list
                                   in coords.iteritems())

    def set_params_filename(self, new=False, start_index=None):
        # if this is not run and params_filename is
        # unspecified, then it won't be passed to gmtkViterbiNew

        params_filename = self.params_filename
        if not new and params_filename:
            if (not self.delete_existing
                and path(params_filename).exists()):
                # it already exists and you don't want to force regen
                self.train = False
        else:
            filebasename = extjoin_not_none(PREFIX_PARAMS, start_index,
                                            EXT_PARAMS)

            self.params_filename = self.dirpath / filebasename

    def set_log_likelihood_filename(self, start_index=None):
        # no need for new=False, since I don't care about keeping this file
        # around generally
        filebasename = extjoin_not_none(PREFIX_LIKELIHOOD, start_index,
                                        EXT_LIKELIHOOD)
        self.log_likelihood_filename = self.dirpath / filebasename

    def make_dir(self, dirname):
        dirpath = path(dirname)

        if self.delete_existing:
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

    def make_wig_dir(self):
        self.make_dir(self.wig_dirname)

    def make_obs_dir(self):
        obs_dirname = self.obs_dirname
        if not obs_dirname:
            self.obs_dirname = obs_dirname = self.dirname

        obs_dirpath = path(obs_dirname)

        try:
            self.make_dir(obs_dirname)
        except OSError, err:
            if not (err.errno == EEXIST and obs_dirpath.isdir()):
                raise

        self.obs_dirpath = obs_dirpath
        self.feature_filelistpath = obs_dirpath / FEATURE_FILELISTBASENAME

    def make_obs_filepath(self, chunk_index):
        prefix_feature_tmpl = PREFIX_CHUNK + make_prefix_fmt(MAX_CHUNKS)
        prefix = prefix_feature_tmpl % chunk_index

        return path(self.obs_dirpath / (prefix + EXT_OBS))

    def save_resource(self, resname):
        orig_filename = data_filename(resname)

        if self.is_dirname_temp:
            return orig_filename
        else:
            orig_filepath = path(orig_filename)
            dirpath = self.dirpath

            orig_filepath.copy(dirpath)
            return dirpath / orig_filepath.name

    def save_include(self):
        self.gmtk_include_filename = self.save_resource(RES_INC)

    def save_structure(self):
        observation_tmpl = Template(data_string("observation.tmpl"))
        observation_sub = observation_tmpl.substitute

        num_obs = self.num_obs
        observations = \
            "\n".join(observation_sub(obs_index=obs_index,
                                      nonmissing_index=num_obs+obs_index)
                      for obs_index in xrange(num_obs))

        mapping = dict(include_filename=self.gmtk_include_filename,
                       observations=observations)

        self.structure_filename = \
            save_template(self.structure_filename, RES_STR_TMPL, mapping,
                          self.dirname, self.delete_existing)

    def save_observations_chunk(self, outfilename, data):
        with open(outfilename, "w") as outfile:
            mask_missing = isnan(data)
            mask_nonmissing = (~ mask_missing).astype(int)

            data[mask_missing] = SENTINEL

            for data_row, mask_nonmissing_row in zip(data, mask_nonmissing):
                ## XXX: use textinput.ListWriter
                outrow = data_row.tolist() + mask_nonmissing_row.tolist()

                print >>outfile, " ".join(map(str, outrow))

    def write_observations(self, feature_filelist):
        make_obs_filepath = self.make_obs_filepath
        save_observations_chunk = self.save_observations_chunk
        delete_existing = self.delete_existing

        num_obs = None
        mins = None
        maxs = None

        chunk_index = 0
        chunk_coords = []

        include_coords = self.include_coords

        for h5filename in self.h5filenames:
            print >>sys.stderr, h5filename
            chrom = path(h5filename).namebase

            if include_coords:
                try:
                    chr_include_coords = include_coords[chrom]
                except KeyError:
                    # nothing is included on that chromosome
                    continue

            with openFile(h5filename) as chromosome:
                try:
                    mins, maxs = accum_extrema(chromosome, mins, maxs)
                except AttributeError:
                    # this means there is no data for that chromosome
                    continue

                tracknames = get_tracknames(chromosome)
                if self.tracknames is None:
                    self.tracknames = tracknames
                elif self.tracknames != tracknames:
                    raise ValueError("all tracknames attributes must be"
                                     " identical")

                supercontig_walker = walk_continuous_supercontigs(chromosome)
                for supercontig, continuous in supercontig_walker:
                    # also asserts same shape
                    num_obs = init_num_obs(num_obs, continuous)

                    supercontig_attrs = supercontig._v_attrs
                    supercontig_start = supercontig_attrs.start

                    convert_chunks_custom = partial(convert_chunks,
                                                    supercontig_attrs)

                    starts = convert_chunks_custom("chunk_starts")
                    ends = convert_chunks_custom("chunk_ends")

                    ## iterate through chunks and write
                    ## izip so it can be modified in place
                    for start, end in izip(starts, ends):
                        if include_coords:
                            overlaps = find_overlaps(start, end,
                                                     chr_include_coords)
                            len_overlaps = len(overlaps)

                            if len_overlaps == 0:
                                continue
                            elif len_overlaps == 1:
                                start, end = overlaps[0]
                            else:
                                for overlap in overlaps:
                                    starts.append(overlap[0])
                                    ends.append(overlap[1])
                                continue

                        num_frames = end - start
                        if not MIN_FRAMES <= num_frames <= MAX_FRAMES:
                            text = " skipping segment of length %d" \
                                % num_frames
                            print >>sys.stderr, text
                            continue

                        # start: reltaive to beginning of chromosome
                        # chunk_start: relative to the beginning of
                        # the supercontig
                        chunk_start = start - supercontig_start
                        chunk_end = end - supercontig_start
                        chunk_coords.append((chrom, start, end))

                        chunk_filepath = make_obs_filepath(chunk_index)

                        print >>feature_filelist, chunk_filepath
                        print >>sys.stderr, " %s (%d, %d)" % (chunk_filepath,
                                                              start, end)

                        if not chunk_filepath.exists():
                            rows = continuous[chunk_start:chunk_end, ...]
                            save_observations_chunk(chunk_filepath, rows)

                        chunk_index += 1

        self.num_obs = num_obs
        self.num_chunks = chunk_index
        self.chunk_coords = chunk_coords
        self.mins = mins
        self.maxs = maxs

    def save_observations(self):
        feature_filelistpath = self.feature_filelistpath

        if self.delete_existing or feature_filelistpath.exists():
            feature_filelist = closing(StringIO()) # dummy output
        else:
            feature_filelist = open(self.feature_filelistpath, "w")

        with feature_filelist as feature_filelist:
            self.write_observations(feature_filelist)

    def save_input_master(self, new=False, start_index=None):
        num_segs = self.num_segs
        num_obs = self.num_obs
        mins = self.mins
        maxs = self.maxs

        include_filename = self.gmtk_include_filename

        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        dense_cpt_spec = make_dense_cpt_spec(num_segs)
        mean_spec = make_mean_spec(num_segs, num_obs, mins, maxs)
        covar_spec = make_covar_spec(num_segs, num_obs, mins, maxs, COVAR_TIED)
        mc_spec = make_mc_spec(num_segs, num_obs)
        mx_spec = make_mx_spec(num_segs, num_obs)
        name_collection_spec = make_name_collection_spec(num_segs, num_obs)

        self.input_master_filename = \
            save_template(input_master_filename, RES_INPUT_MASTER_TMPL,
                          locals(), self.dirname, self.delete_existing,
                          start_index)
    def save_dont_train(self):
        self.dont_train_filename = self.save_resource(RES_DONT_TRAIN)

    def save_output_filelist(self):
        dirpath = self.dirpath
        num_chunks = self.num_chunks

        output_filename_fmt = "out" + make_prefix_fmt(num_chunks) + EXT_OUT
        output_filenames = [dirpath / output_filename_fmt % index
                            for index in xrange(num_chunks)]

        output_filelistname = dirpath / extjoin("output", EXT_LIST)
        self.output_filelistname = output_filelistname

        with open(output_filelistname, "w") as output_filelist:
            for output_filename in output_filenames:
                print >>output_filelist, output_filename

        self.output_filenames = output_filenames

    def save_dumpnames(self):
        dirpath = self.dirpath
        dumpnames_filename = dirpath / extjoin("dumpnames", EXT_LIST)
        self.dumpnames_filename = dumpnames_filename

        with open(dumpnames_filename, "w") as dumpnames_file:
            print >>dumpnames_file, "seg"

    def save_params(self):
        self.load_include_coords()

        self.make_obs_dir()
        self.save_observations() # do first, because it sets self.num_obs

        self.save_include()
        self.save_structure()

        if self.train:
            self.save_dont_train()
            self.set_params_filename() # might turn off self.train
            self.set_log_likelihood_filename()

        if self.identify:
            self.save_output_filelist()
            self.save_dumpnames()
            self.make_wig_dir()

    def move_results(self, name, src_filename, dst_filename):
        if dst_filename:
            move(src_filename, dst_filename)
        else:
            dst_filename = src_filename

        setattr(self, name, dst_filename)

    def gmtk_out2wig(self):
        prefix_fmt = make_prefix_fmt(self.num_chunks)
        wig_filebasename_fmt = "".join([PKG, prefix_fmt, EXT_WIG, SUFFIX_GZ])

        wig_dirpath = path(self.wig_dirname)
        wig_filepath_fmt = wig_dirpath / wig_filebasename_fmt

        # chunk_coord = (chrom, chromStart, chromEnd)
        zipper = izip(count(), self.output_filenames, self.chunk_coords)
        for index, gmtk_outfilename, chunk_coord in zipper:
            wig_filename = wig_filepath_fmt % index

            load_gmtk_out_save_wig(chunk_coord, gmtk_outfilename,
                                   wig_filename, self.tracknames)

    def prog_factory(self, prog):
        """
        allows dry_run
        """
        def dry_run_prog(*args, **kwargs):
            print " ".join(prog.build_cmdline(args, kwargs))

        if self.dry_run:
            return dry_run_prog
        else:
            return prog

    def queue_parallel(self, session, params_filename, start_index,
                       round_index, **kwargs):
        kwargs = dict(inputMasterFile=self.input_master_filename,
                      inputTrainableParameters=params_filename,
                      cppCommandOptions=make_cpp_options(params_filename),
                      **kwargs)

        res = [] # task ids

        # sort chunks by decreasing size, so the most difficult chunks
        # are dropped in the queue first
        zipper = izip(self.chunk_lens, count(), self.chunk_mem_reqs)
        sorted_zipper = sorted(zipper, reverse=True)

        for chunk_len, chunk_index, chunk_mem_req in sorted_zipper:
            # XXX: there is some duplication between this and
            # queue_bundle, which should be removed. refactor the
            # per-chunk stuff into a new function

            acc_filebasename = ACC_FILENAME_FMT % (start_index, chunk_index)
            acc_filename = self.dirpath / acc_filebasename
            kwargs_chunk = dict(trrng=chunk_index,
                                storeAccFile=acc_filename,
                                **kwargs)

            gmtk_cmdline = self.train_prog.build_cmdline(options=kwargs_chunk)

            job_tmpl = session.createJobTemplate()

            name_job(job_tmpl, start_index, round_index, chunk_index)
            job_tmpl.remoteCommand = ENV_CMD
            job_tmpl.args = EM_TRAIN_CMDLINE + gmtk_cmdline

            set_cwd_job_tmpl(job_tmpl)

            res_req = RES_REQ_FMT % chunk_mem_req
            job_tmpl.nativeSpecification = make_native_spec(l=res_req)

            res.append(session.runJob(job_tmpl))

        return res

    def queue_bundle(self, session, parallel_jobids, input_params_filename,
                     output_params_filename, start_index, round_index,
                     **kwargs):
        ## bundle step: take parallel accumulators and combine them
        acc_filebasename = ACC_FILENAME_FMT % (start_index,
                                               GMTK_INDEX_PLACEHOLDER)
        acc_filename = self.dirpath / acc_filebasename

        kwargs = \
            dict(inputMasterFile=self.input_master_filename,
                 inputTrainableParameters=input_params_filename,
                 outputTrainableParameters=output_params_filename,
                 cppCommandOptions=make_cpp_options(input_params_filename),
                 trrng="nil",
                 loadAccRange="0:%s" % (self.num_chunks-1),
                 loadAccFile=acc_filename,
                 **kwargs)

        gmtk_cmdline = self.train_prog.build_cmdline(options=kwargs)
        job_tmpl = session.createJobTemplate()

        name_job(job_tmpl, start_index, round_index, "bundle")
        job_tmpl.remoteCommand = ENV_CMD
        job_tmpl.args = EM_TRAIN_CMDLINE + gmtk_cmdline
        set_cwd_job_tmpl(job_tmpl)

        res_req = RES_REQ_FMT % MEM_REQ_BUNDLE
        job_tmpl.nativeSpecification = \
            make_native_spec(hold_jid=",".join(parallel_jobids), l=res_req)

        return session.runJob(job_tmpl)

    def run_triangulate(self):
        # XXX: should specify the triangulation file
        prog = self.prog_factory(TRIANGULATE_PROG)

        prog(strFile=self.structure_filename,
             verbosity=VERBOSITY)

    def run_train_start(self, start_index):
        # XXX: re-add the ability to set your own starting parameters,
        # with new=start_index (copy from existing rather than using
        # it on command-line)
        self.save_input_master(new=True, start_index=start_index)
        self.set_params_filename(new=True, start_index=start_index)
        self.set_log_likelihood_filename(start_index=start_index)

        log_likelihood_filename = self.log_likelihood_filename
        last_log_likelihood = NINF
        log_likelihood = NINF
        round_index = 0

        stem_params_filename = self.params_filename
        last_params_filename = None
        curr_params_filename = None

        kwargs = self.train_kwargs

        queue_parallel = self.queue_parallel
        queue_bundle = self.queue_bundle

        with Session() as session:
            while (round_index < MAX_EM_ITERS
                   and is_training_progressing(last_log_likelihood,
                                               log_likelihood)):
                parallel_jobids = queue_parallel(session, last_params_filename,
                                                 start_index, round_index,
                                                 **kwargs)

                curr_params_filename = extjoin(stem_params_filename,
                                               str(round_index))

                bundle_jobid = queue_bundle(session, parallel_jobids,
                                            last_params_filename,
                                            curr_params_filename,
                                            start_index, round_index,
                                            llStoreFile=\
                                                log_likelihood_filename,
                                            **kwargs)

                # wait for bundle to finish
                session.wait(bundle_jobid, session.TIMEOUT_WAIT_FOREVER)

                last_log_likelihood = log_likelihood
                log_likelihood = self.load_log_likelihood()

                print >>sys.stderr, "log likelihood = %s" % log_likelihood

                last_params_filename = curr_params_filename

                round_index += 1

        return log_likelihood, self.input_master_filename, last_params_filename

    def run_train(self):
        assert not self.dry_run

        self.train_prog = self.prog_factory(EM_TRAIN_PROG)

        self.train_kwargs = dict(strFile=self.structure_filename,
                                 objsNotToTrain=self.dont_train_filename,

                                 of1=self.feature_filelistpath,
                                 fmt1="ascii",
                                 nf1=self.num_obs,
                                 ni1=self.num_obs,

                                 maxEmIters=1,
                                 verbosity=VERBOSITY,
                                 island=ISLAND,
                                 componentCache=COMPONENT_CACHE,
                                 lldp=LOG_LIKELIHOOD_DIFF_FRAC*100.0)

        # save the destination file for input_master as we will be
        # generating new input masters for each start
        dst_filenames = [self.input_master_filename,
                         self.params_filename]

        chunk_lens = [end - start for chr, start, end in self.chunk_coords]
        self.chunk_lens = chunk_lens
        self.chunk_mem_reqs = [make_mem_req(len) for len in chunk_lens]

        # list of tuples(log_likelihood, input_master_filename,
        #                params_filename)
        start_params = []
        for start_index in xrange(self.random_starts):
            # keeps it from rewriting variables that will be used
            # later or in a different thread
            runner_copy = copy(self)
            start_params.append(runner_copy.run_train_start(start_index))

        if not self.dry_run:
            src_filenames = max(start_params)[1:]

            zipper = zip(TRAIN_ATTRNAMES, src_filenames, dst_filenames)
            for name, src_filename, dst_filename in zipper:
                self.move_results(name, src_filename, dst_filename)

    def run_identify(self):
        if not self.input_master_filename:
            self.save_input_master()

        params_filename = self.params_filename

        prog = self.prog_factory(VITERBI_PROG)

        prog(strFile=self.structure_filename,

             inputMasterFile=self.input_master_filename,
             inputTrainableParameters=params_filename,

             ofilelist=self.output_filelistname,
             dumpNames=self.dumpnames_filename,

             of1=self.feature_filelistpath,
             fmt1="ascii",
             nf1=self.num_obs,
             ni1=self.num_obs,

             cppCommandOptions=make_cpp_options(params_filename),
             verbosity=VERBOSITY)

        if not self.dry_run:
            self.gmtk_out2wig()

    def _run(self):
        """
        main run, after dirname is specified
        """
        # XXX: use binary I/O to gmtk rather than ascii

        self.dirpath = path(self.dirname)
        self.save_params()

        if self.triangulate:
            self.run_triangulate()

        # XXX: make tempfile to specify for -jtFile for both
        # em and viterbi
        if self.train:
            self.run_train()

        if self.identify:
            self.run_identify()

    def __call__(self):
        # XXX: register atexit for cleanup_resources

        dirname = self.dirname
        if dirname:
            if self.delete_existing or not path(dirname).isdir():
                self.make_dir(dirname)

            self._run()
        else:
            try:
                with NamedTemporaryDir(prefix=TEMPDIR_PREFIX) as tempdir:
                    self.dirname = tempdir.name
                    self.is_dirname_temp = True
                    self._run()
            finally:
                # the temporary directory has already been deleted (after
                # unwinding of the with block), so let's stop referring to
                # it
                self.dirname = None
                self.is_dirname_temp = False

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... H5FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    # XXX: group here: filenames
    parser.add_option("--observations", "-o", metavar="DIR",
                      help="use or create observations in DIR")

    parser.add_option("--wiggle", "-w", metavar="DIR",
                      help="use or create wiggle tracks in DIR",
                      default="out")

    parser.add_option("--input-master", "-i", metavar="FILE",
                      help="use or create input master in FILE")

    parser.add_option("--structure", "-s", metavar="FILE",
                      help="use or create structure in FILE")

    parser.add_option("--trainable-params", "-p", metavar="FILE",
                      help="use or create trainable parameters in FILE")

    parser.add_option("--directory", "-d", metavar="DIR",
                      help="create all other files in DIR")

    # this is a 0-based file (I know because ENm008 starts at position 0)
    parser.add_option("--include-coords", metavar="FILE",
                      help="limit to genomic coordinates in FILE")

    # XXX: group here: variables
    parser.add_option("--random-starts", "-r", type=int, default=RANDOM_STARTS,
                      metavar="NUM",
                      help="randomize start parameters NUM times")

    # XXX: group here: flag options
    parser.add_option("--force", "-f", action="store_true",
                      help="delete any preexisting files")
    parser.add_option("--no-identify", "-I", action="store_true",
                      help="do not identify segments")
    parser.add_option("--no-train", "-T", action="store_true",
                      help="do not train model")
    parser.add_option("--dry-run", "-n", action="store_true",
                      help="write all files, but do not run any executables")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)
    dirname = options.directory
    wig_dirname = options.wiggle

    if dirname and not wig_dirname:
        wig_dirname = path(dirname) / WIG_DIRNAME

    runner = Runner()

    runner.h5filenames = args
    runner.dirname = dirname
    runner.obs_dirname = options.observations
    runner.wig_dirname = wig_dirname
    runner.input_master_filename = options.input_master
    runner.structure_filename = options.structure
    runner.params_filename = options.trainable_params
    runner.include_coords_filename = options.include_coords

    runner.random_starts = options.random_starts

    runner.delete_existing = options.force
    runner.train = not options.no_train
    runner.identify = not options.no_identify
    runner.dry_run = options.dry_run

    return runner()

if __name__ == "__main__":
    sys.exit(main())
