#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from cStringIO import StringIO
from contextlib import closing, contextmanager, nested
from copy import copy
from errno import EEXIST, ENOENT
from functools import partial
from itertools import count, izip
from math import ceil, floor, ldexp, log10
from os import extsep, getpid
from shutil import move
from string import Template
import sys
from threading import Event, Thread

from DRMAA import ExitTimeoutError, Session as _Session
from numpy import (amin, amax, array, diag, diff, empty, finfo,
                   float32, fromfile, invert, isnan, newaxis, NINF, s_,
                   where)
from numpy.random import uniform
from optbuild import (Mixin_NoConvertUnderscore,
                      OptionBuilder_ShortOptWithSpace,
                      OptionBuilder_ShortOptWithSpace_TF)
from tables import Atom, openFile
from path import path

from ._util import (data_filename, data_string, DTYPE_IDENTIFY, DTYPE_OBS_INT,
                    DTYPE_SEG_LEN, EXT_GZ, fill_array, find_segment_starts,
                    FILTERS_GZIP, get_tracknames, gzip_open,
                    iter_chroms_coords, load_coords,
                    NamedTemporaryDir, PKG,
                    walk_continuous_supercontigs)

DISTRIBUTION_NORM = "norm"
DISTRIBUTION_GAMMA = "gamma"

## XXX: should be options
NUM_SEGS = 2 # XXX: will require CARD_SEG to be set
MAX_EM_ITERS = 100
VERBOSITY = 0 # XXX: should vary based on DRMAA submission or not
TEMPDIR_PREFIX = PKG + "-"
COVAR_TIED = True # would need to expand to MC, MX to change
MAX_CHUNKS = 1000
ISLAND = False
BED_DIRNAME = "out"
SESSION_WAIT_TIMEOUT = 60 # seconds
JOIN_TIMEOUT = finfo(float).max
LEN_SEG_EXPECTED = 10000

DRMSINFO_PREFIX = "GE" # XXX: only SGE is supported for now

FINFO_FLOAT32 = finfo(float32)
MACHEP_FLOAT32 = FINFO_FLOAT32.machep
TINY_FLOAT32 = FINFO_FLOAT32.tiny

FUDGE_EP = -17 # ldexp(1, -17) = ~1e-6
assert FUDGE_EP > MACHEP_FLOAT32

FUDGE_TINY = -ldexp(TINY_FLOAT32, 6)

DISTRIBUTION = DISTRIBUTION_GAMMA

LOG_LIKELIHOOD_DIFF_FRAC = 1e-5

# for extra memory savings, set to (False) or (not ISLAND)
COMPONENT_CACHE = True

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2
MAX_FRAMES = 1000000000 # 1 billion
MEM_REQ_PARALLEL = "10.5G"
MEM_REQ_BUNDLE = "500M"
RES_REQ_IDS = ["mem_requested", "mem_free"]

# for a four-way model
MEM_REQS = {1: [3619, 8098728],
            2: [3619, 8098728],
            3: [1553, 16121352],
            4: [5768, 14442884]}

## defaults
RANDOM_STARTS = 1

# replace NAN with SENTINEL to avoid warnings
# XXX: replace with something negative and outlandish again
SENTINEL = float32(9.87654321)

ACC_FILENAME_FMT = "acc.%s.%s.bin"
GMTK_INDEX_PLACEHOLDER = "@D"
NAME_PLACEHOLDER = "bundle"

# programs
ENV_CMD = "/usr/bin/env"
BASH_CMD = "bash"

BASH_CMDLINE = [BASH_CMD, "--login", "-c"]

TRIANGULATE_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkEMtrainNew")
VITERBI_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkViterbiNew")
NATIVE_SPEC_PROG = (Mixin_NoConvertUnderscore
                    + OptionBuilder_ShortOptWithSpace)() # do not run

NATIVE_SPEC_DEFAULT = dict(w="n")

OPT_USE_TRAINABLE_PARAMS = "-DUSE_TRAINABLE_PARAMS"

def extjoin(*args):
    return extsep.join(args)

# extensions and suffixes
EXT_BED = "bed"
EXT_IDENTIFY = "identify.h5"
EXT_LIKELIHOOD = "ll"
EXT_LIST = "list"
EXT_FLOAT = "float32"
EXT_INT = "int"
EXT_OUT = "out"
EXT_PARAMS = "params"
EXT_TAB = "tab"

def make_prefix_fmt(num):
    # make sure there are sufficient leading zeros
    return "%%0%dd." % (int(floor(log10(num))) + 1)

PREFIX_LIKELIHOOD = "likelihood"
PREFIX_CHUNK = "chunk"
PREFIX_PARAMS = "params"

# XXX: do not hardcode
PREFIX_SEG_LEN_FMT = "seg%s" % make_prefix_fmt(NUM_SEGS)

SUFFIX_LIST = extsep + EXT_LIST
SUFFIX_OUT = extsep + EXT_OUT

IDENTIFY_FILELISTBASENAME = extjoin("identify", EXT_LIST)

# templates and formats
RES_STR_TMPL = "seg.str.tmpl"
RES_INPUT_MASTER_TMPL = "input.master.tmpl"
RES_DONT_TRAIN = "dont_train.list"
RES_INC = "seg.inc"

DIRICHLET_FRAG = "0 dirichlet_seg_seg 2 CARD_SEG CARD_SEG"

DENSE_CPT_START_SEG_FRAG = "0 start_seg 0 CARD_SEG"
DENSE_CPT_SEG_SEG_FRAG = "1 seg_seg 1 CARD_SEG CARD_SEG"
DIRICHLET_SEG_SEG_FRAG = "DirichletTable dirichlet_seg_seg"

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

# XXX: this could be specified as a dict instead
WIG_HEADER = 'track type=wiggle_0 name=%s ' \
    'description="%s segmentation of %%s" visibility=dense viewLimits=0:1 ' \
    'autoScale=off' % (PKG, PKG)

TRAIN_ATTRNAMES = ["input_master_filename", "params_filename",
                   "log_likelihood_filename"]

def extjoin_not_none(*args):
    return extjoin(*[str(arg) for arg in args
                     if arg is not None])

# XXX: suggest upstream as addition to DRMAA-python
@contextmanager
def Session(*args, **kwargs):
    res = _Session()
    res.init(*args, **kwargs)

    assert res.DRMSInfo.startswith(DRMSINFO_PREFIX)

    try:
        yield res
    finally:
        res.exit()

def make_res_req(size):
    res = []
    for res_req_id in RES_REQ_IDS:
        res.append("%s=%s" % (res_req_id, size))

    return res

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

def make_mem_req(len, num_tracks):
    # will fail if it's not pre-defined
    slope, intercept = MEM_REQS[num_tracks]
    res = slope * len + intercept

    return "%dM" % ceil(res / 2**20)

def make_cpp_options(params_filename):
    if params_filename:
        return OPT_USE_TRAINABLE_PARAMS
    else:
        return None

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

def make_normalized_random_table(num_rows, num_cols):
    random_vals = uniform(size=(num_rows, num_cols))
    random_vals_sums = random_vals.sum(1)[..., newaxis]

    return random_vals / random_vals_sums

def make_random_spec(frag, *args, **kwargs):
    table = make_normalized_random_table(*args, **kwargs)

    return make_table_spec(frag, table)

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

def write_segment_summary_stats(start_pos, labels, seg_len_files):
    # XXX: should use HDF5 output instead

    for seg_index, seg_len_file in enumerate(seg_len_files):
        where_seg, = where(labels == seg_index)
        coords_seg = start_pos.take([where_seg, where_seg+1])

        lens_seg = diff(coords_seg, axis=0).ravel()

        lens_seg.astype(DTYPE_SEG_LEN).tofile(seg_len_file)

def load_gmtk_out(filename):
    # gmtkViterbiNew.cc writes things with C sizeof(int) == numpy.intc
    return fromfile(filename, dtype=DTYPE_IDENTIFY)

def write_identify(h5file, data, chrom, start, end, tracknames):
    root = h5file.root
    attrs = root._v_attrs

    attrs.chrom = chrom
    attrs.start = start
    attrs.end = end
    attrs.tracknames = array(tracknames)

    # this is slower than just using a tables.Array, but it allows compression
    atom = Atom.from_dtype(data.dtype)
    c_array = h5file.createCArray(root, "identify", atom, data.shape)

    c_array[...] = data

def write_bed(outfile, start_pos, labels, coords, tracknames):
    # XXX: add in optional browser track line (see SVN revisions
    # previous to 195)

    (chrom, region_start, region_end) = coords

    start_pos += region_start

    zipper = zip(start_pos[:-1], start_pos[1:], labels)
    for seg_start, seg_end, seg_label in zipper:
        row = [chrom, str(seg_start), str(seg_end), str(seg_label)]
        print >>outfile, "\t".join(row)

def load_gmtk_out_write_bed((chrom, start, end), gmtk_outfilename,
                            identify_filename, bed_file, seg_len_files,
                            tracknames):
    data = load_gmtk_out(gmtk_outfilename)

    identify_file = openFile(identify_filename, "w", chrom,
                             filters=FILTERS_GZIP)
    with identify_file:
        write_identify(identify_file, data, chrom, start, end, tracknames)

    start_pos, labels = find_segment_starts(data)

    write_bed(bed_file, start_pos, labels, (chrom, start, end), tracknames)
    write_segment_summary_stats(start_pos, labels, seg_len_files)

def set_cwd_job_tmpl(job_tmpl):
    job_tmpl.workingDirectory = path.getcwd()

def generate_tmpl_mappings(segnames, tracknames):
    num_tracks = len(tracknames)

    for seg_index, segname in enumerate(segnames):
        for track_index, trackname in enumerate(tracknames):
            yield dict(seg=segname, track=trackname,
                       seg_index=seg_index, track_index=track_index,
                       index=num_tracks*seg_index + track_index,
                       distribution=DISTRIBUTION)

class RandomStartThread(Thread):
    def __init__(self, runner, session, start_index, interrupt_event):
        # keeps it from rewriting variables that will be used
        # later or in a different thread
        self.runner = copy(runner)

        self.session = session
        self.start_index = start_index
        self.interrupt_event = interrupt_event

        Thread.__init__(self)

    def run(self):
        self.result = self.runner.run_train_start(self.session,
                                                  self.start_index,
                                                  self.interrupt_event)

class Runner(object):
    def __init__(self, **kwargs):
        # filenames
        self.h5filenames = None
        self.float_filelistpath = None
        self.int_filelistpath = None

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
        self.bed_dirname = None

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
        self.segnames = ["seg%d" % seg_index for seg_index in xrange(NUM_SEGS)]
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

        self.include_coords = load_coords(filename)

    def make_filename(self, *exts):
        filebasename = extjoin_not_none(*exts)

        return self.dirpath / filebasename

    def set_tracknames(self, chromosome):
        # XXXopt: a lot of stuff here repeated for every chromosome
        # unnecessarily
        tracknames = get_tracknames(chromosome)
        include_tracknames = set(self.include_tracknames)

        if include_tracknames:
            indexed_tracknames = ((index, trackname)
                                  for index, trackname in enumerate(tracknames)
                                  if trackname in include_tracknames)
            track_indexes, tracknames = zip(*indexed_tracknames)
            track_indexes = array(track_indexes)

        # replace illegal characters
        tracknames = [trackname.replace(".", "_") for trackname in tracknames]

        if self.tracknames is None:
            self.tracknames = tracknames
            self.track_indexes = track_indexes
        elif (self.tracknames != tracknames
              or (self.track_indexes != track_indexes).any()):
            raise ValueError("all tracknames attributes must be identical")

        return track_indexes

    def set_params_filename(self, start_index=None, new=False):
        # if this is not run and params_filename is
        # unspecified, then it won't be passed to gmtkViterbiNew

        params_filename = self.params_filename
        if not new and params_filename:
            if (not self.delete_existing
                and path(params_filename).exists()):
                # it already exists and you don't want to force regen
                self.train = False
        else:
            self.params_filename = \
                self.make_filename(PREFIX_PARAMS, start_index, EXT_PARAMS)

    def set_log_likelihood_filename(self, start_index=None, new=False):
        if new or not self.log_likelihood_filename:
            self.log_likelihood_filename = \
                self.make_filename(PREFIX_LIKELIHOOD, start_index,
                                   EXT_LIKELIHOOD)

    def make_output_dirpath(self, dirname, start_index):
        res = self.dirpath / "output" / dirname / str(start_index)
        self.make_dir(res)

        return res

    def set_output_dirpaths(self, start_index):
        self.output_dirpath = self.make_output_dirpath("o", start_index)
        self.error_dirpath = self.make_output_dirpath("e", start_index)

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

    def make_bed_dir(self):
        self.make_dir(self.bed_dirname)

    def make_obs_filelistpath(self, ext):
        return self.obs_dirpath / extjoin(ext, EXT_LIST)

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

        self.float_filelistpath = self.make_obs_filelistpath(EXT_FLOAT)
        self.int_filelistpath = self.make_obs_filelistpath(EXT_INT)

    def make_obs_filepath(self, prefix, suffix):
        return self.obs_dirpath / (prefix + suffix)

    def make_obs_filepaths(self, chrom, chunk_index):
        prefix_feature_tmpl = extjoin(chrom, make_prefix_fmt(MAX_CHUNKS))
        prefix = prefix_feature_tmpl % chunk_index

        make_obs_filepath_custom = partial(self.make_obs_filepath, prefix)

        return (make_obs_filepath_custom(EXT_FLOAT),
                make_obs_filepath_custom(EXT_INT))

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

        tracknames = self.tracknames
        num_tracks = self.num_tracks
        observations = \
            "\n".join(observation_sub(track=track,
                                      track_index=track_index,
                                      nonmissing_index=num_tracks+track_index)
                      for track_index, track in enumerate(tracknames))

        mapping = dict(include_filename=self.gmtk_include_filename,
                       observations=observations)

        self.structure_filename, self.structure_filename_new = \
            save_template(self.structure_filename, RES_STR_TMPL, mapping,
                          self.dirname, self.delete_existing)

    def save_observations_chunk(self, float_filepath, int_filepath, data):
        # input function in GMTK_ObservationMatrix.cc:
        # ObservationMatrix::readBinSentence

        # input per frame is a series of float32s, followed by a series of
        # int32s it is better to optimize both sides here by sticking all
        # the floats in one file, and the ints in another one
        mask_missing = isnan(data)
        mask_nonmissing = empty(mask_missing.shape, DTYPE_OBS_INT)

        # output -> mask_nonmissing
        invert(mask_missing, mask_nonmissing)

        data[mask_missing] = SENTINEL

        data.tofile(float_filepath)
        mask_nonmissing.tofile(int_filepath)

    def write_observations(self, float_filelist, int_filelist):
        include_coords = self.include_coords

        # originally, the metadata and observations parts were two
        # separate passes, but this is no longer needed. the comment
        # is for convenience in case I need to go back to two separate
        # passes

        # metadata
        mins = None
        maxs = None

        # observations
        make_obs_filepaths = self.make_obs_filepaths
        save_observations_chunk = self.save_observations_chunk
        delete_existing = self.delete_existing

        num_tracks = None # this is before any subsetting
        chunk_index = 0
        chunk_coords = []
        num_bases = 0

        chrom_iterator = iter_chroms_coords(self.h5filenames, include_coords)
        for chrom, filename, chromosome, chr_include_coords in chrom_iterator:
            assert not chromosome.root._v_attrs.dirty

            # metadata
            try:
                mins, maxs = accum_extrema(chromosome, mins, maxs)
            except AttributeError:
                # this means there is no data for that chromosome
                continue

            track_indexes = self.set_tracknames(chromosome)
            num_tracks = len(track_indexes)

            # observations
            supercontig_walker = walk_continuous_supercontigs(chromosome)
            for supercontig, continuous in supercontig_walker:
                assert continuous.shape[1] >= num_tracks

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
                        text = " skipping segment of length %d" % num_frames
                        print >>sys.stderr, text
                        continue

                    # start: relative to beginning of chromosome
                    # chunk_start: relative to the beginning of
                    # the supercontig
                    chunk_start = start - supercontig_start
                    chunk_end = end - supercontig_start
                    chunk_coords.append((chrom, start, end))

                    float_filepath, int_filepath = \
                        make_obs_filepaths(chrom, chunk_index)

                    print >>float_filelist, float_filepath
                    print >>int_filelist, int_filepath
                    print >>sys.stderr, " %s (%d, %d)" % (float_filepath,
                                                          start, end)

                    num_bases += end - start

                    # if they don't both exist
                    if not (float_filepath.exists() and int_filepath.exists()):
                        # read rows first into a numpy.array because
                        # you can't do complex imports on a
                        # numpy.Array
                        min_col = track_indexes.min()
                        max_col = track_indexes.max() + 1
                        col_slice = s_[min_col:max_col]

                        rows = continuous[chunk_start:chunk_end, col_slice]

                        # correct for min_col offset
                        cells = rows[..., track_indexes - min_col]

                        save_observations_chunk(float_filepath, int_filepath,
                                                cells)

                    chunk_index += 1

        self.mins = mins
        self.maxs = maxs

        self.num_tracks = num_tracks
        self.num_chunks = chunk_index
        self.num_bases = num_bases
        self.chunk_coords = chunk_coords

    def open_writable_or_dummy(self, filepath):
        if self.delete_existing or filepath.exists():
            return closing(StringIO()) # dummy output
        else:
            return open(filepath, "w")

    def save_observations(self):
        open_writable_or_dummy = self.open_writable_or_dummy

        with open_writable_or_dummy(self.float_filelistpath) as float_filelist:
            with open_writable_or_dummy(self.int_filelistpath) as int_filelist:
                self.write_observations(float_filelist, int_filelist)

    def rand_means(self):
        low = self.mins
        high = self.maxs
        num_segs = self.num_segs

        assert len(low) == len(high)

        # size parameter is so that we always get an array, even if it
        # has shape = (1,)
        return array([uniform(low, high, len(low))
                      for seg_index in xrange(num_segs)])

    def make_items_multiseg(self, tmpl, data=None, segnames=None):
        tracknames = self.tracknames
        mins = self.mins

        if segnames is None:
            segnames = self.segnames

        substitute = Template(tmpl).substitute

        res = []
        for mapping in generate_tmpl_mappings(segnames, tracknames):
            track_index = mapping["track_index"]
            min_track = mins[track_index]

            # fudge the minimum by a very small amount this is not
            # continuous, but hopefully we won't get values where it
            # matters
            # XXX: restore this after GMTK issues fixed
#            if min_track == 0.0:
#                min_track_fudged = FUDGE_TINY
#            else:
#                min_track_fudged = min_track - ldexp(abs(min_track), FUDGE_EP)
#
            # this happens for really big numbers or really small
            # numbers; you only have 7 orders of magnitude to play
            # with on a float32
            assert float32(min_track) - float32(1.0) != float32(min_track)
            mapping["min_track"] = min_track - 1.0

            if data is not None:
                seg_index = mapping["seg_index"]
                mapping["rand"] = data[seg_index, track_index]

            res.append(substitute(mapping))

        return res

    def make_spec_multiseg(self, name, *args, **kwargs):
        return make_spec(name, self.make_items_multiseg(*args, **kwargs))

    def make_dirichlet_table(self):
        num_segs = self.num_segs

        prob_diag = prob_transition_from_expected_len(LEN_SEG_EXPECTED)
        prob_nondiag = (1.0 - prob_diag) / (num_segs - 1)

        probs = diag(fill_array(prob_diag, num_segs))
        probs[probs == 0.0] = prob_nondiag

        # astype(int) means flooring the floats
        total_pseudocounts = self.len_seg_strength * self.num_bases
        pseudocounts_per_row = total_pseudocounts / num_segs
        pseudocounts = (probs * pseudocounts_per_row).astype(int)

        return pseudocounts

    def make_dirichlet_spec(self):
        dirichlet_table = self.make_dirichlet_table()
        items = [make_table_spec(DIRICHLET_FRAG, dirichlet_table)]

        return make_spec("DIRICHLET_TAB", items)

    def make_dense_cpt_start_seg_spec(self):
        return make_random_spec(DENSE_CPT_START_SEG_FRAG, 1, self.num_segs)

    def make_dense_cpt_seg_seg_spec(self):
        num_segs = self.num_segs

        if self.len_seg_strength > 0:
            frag = "\n".join([DENSE_CPT_SEG_SEG_FRAG, DIRICHLET_SEG_SEG_FRAG])
        else:
            frag = DENSE_CPT_SEG_SEG_FRAG

        return make_random_spec(frag, num_segs, num_segs)

    def make_dense_cpt_spec(self):
        num_segs = self.num_segs

        items = [self.make_dense_cpt_start_seg_spec(),
                 self.make_dense_cpt_seg_seg_spec()]

        return make_spec("DENSE_CPT", items)

    def make_mean_spec(self, means):
        return self.make_spec_multiseg("MEAN", MEAN_TMPL, means)

    def make_covar_spec(self, tied, vars):
        if tied:
            segnames = ["any"]
            tmpl = COVAR_TMPL_TIED
        else:
            segnames = None
            tmpl = COVAR_TMPL_UNTIED

        # always start with maximum variance

        return self.make_spec_multiseg("COVAR", tmpl, vars, segnames)

    def make_items_gamma(self, means, vars):
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
        for mapping in generate_tmpl_mappings(self.segnames, self.tracknames):
            seg_index = mapping["seg_index"]
            track_index = mapping["track_index"]
            index = mapping["index"] * 2

            mapping_plus = partial(dict, **mapping)

            cell_indices = (seg_index, track_index)
            scale = scales[cell_indices]
            shape = shapes[cell_indices]

            mapping_scale = mapping_plus(rand=scale, index=index)
            res.append(substitute_scale(mapping_scale))

            mapping_shape = mapping_plus(rand=shape, index=index+1)
            res.append(substitute_shape(mapping_shape))

        return res

    def make_gamma_spec(self, *args, **kwargs):
        return make_spec("REAL_MAT", self.make_items_gamma(*args, **kwargs))

    def make_mc_spec(self):
        return self.make_spec_multiseg("MC", MC_TMPLS[DISTRIBUTION])

    def make_mx_spec(self):
        return self.make_spec_multiseg("MX", MX_TMPL)

    def save_input_master(self, start_index=None, new=False):
        tracknames = self.tracknames
        num_segs = self.num_segs
        mins = self.mins
        maxs = self.maxs

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

        means = self.rand_means()
        vars = array([maxs - mins for seg_index in xrange(num_segs)])

        if DISTRIBUTION == DISTRIBUTION_NORM:
            mean_spec = self.make_mean_spec(means)
            covar_spec = self.make_covar_spec(COVAR_TIED, vars)
            gamma_spec = ""
        elif DISTRIBUTION == DISTRIBUTION_GAMMA:
            mean_spec = ""
            covar_spec = ""
            gamma_spec = self.make_gamma_spec(means, vars)
        else:
            raise ValueError("distribution %s not supported" % DISTRIBUTION)

        mc_spec = self.make_mc_spec()
        mx_spec = self.make_mx_spec()
        name_collection_spec = make_name_collection_spec(num_segs, tracknames)

        self.input_master_filename, self.input_master_filename_new = \
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

        # do first, because it sets self.num_tracks and self.tracknames
        self.save_observations()

        self.save_include()
        self.save_structure()

        train = self.train
        identify = self.identify

        if train or identify:
            self.make_chunk_mem_reqs()

        if train:
            self.save_dont_train()
            self.set_params_filename() # might turn off self.train
            self.set_log_likelihood_filename()

        if identify:
            self.save_output_filelist()
            self.save_dumpnames()
            self.make_bed_dir()

    def move_results(self, name, src_filename, dst_filename):
        if dst_filename:
            move(src_filename, dst_filename)
        else:
            dst_filename = src_filename

        setattr(self, name, dst_filename)

    def gmtk_out2bed(self):
        bed_filebasename = extjoin(PKG, EXT_BED, EXT_GZ)

        prefix_fmt = make_prefix_fmt(self.num_chunks)
        identify_filebase_fmt_list = [PKG, prefix_fmt, EXT_IDENTIFY]
        identify_filebasename_fmt = "".join(identify_filebase_fmt_list)

        out_dirpath = path(self.bed_dirname)
        bed_filepath = out_dirpath / bed_filebasename
        identify_filepath_fmt = out_dirpath / identify_filebasename_fmt

        seg_len_filebasename_fmt = "".join([PREFIX_SEG_LEN_FMT, EXT_INT])
        seg_len_filepath_fmt = out_dirpath / seg_len_filebasename_fmt

        seg_len_filenames = [seg_len_filepath_fmt % seg_index
                             for seg_index in xrange(NUM_SEGS)]

        identify_filelistname = out_dirpath / IDENTIFY_FILELISTBASENAME
        identify_filelist = open(identify_filelistname, "w")

        # XXX: this should be gzipped
        open_wb = partial(open, mode="wb")

        # chunk_coord = (chrom, chromStart, chromEnd)
        zipper = izip(count(), self.output_filenames, self.chunk_coords)
        with nested(*map(open_wb, seg_len_filenames)) as seg_len_files:
            with gzip_open(bed_filepath, "w") as bed_file:
                for index, gmtk_outfilename, chunk_coord in zipper:
                    identify_filename = identify_filepath_fmt % index

                    print >>identify_filelist, identify_filename
                    load_gmtk_out_write_bed(chunk_coord, gmtk_outfilename,
                                           identify_filename, bed_file,
                                           seg_len_files, self.tracknames)

    def prog_factory(self, prog):
        """
        allows dry_run
        """
        # XXX: this poisons a global variable
        prog.dry_run = self.dry_run

        return prog

    def make_acc_filename(self, start_index, chunk_index):
        filebasename = ACC_FILENAME_FMT % (start_index, chunk_index)
        return self.dirpath / filebasename

    def make_job_name_train(self, start_index, round_index, chunk_index):
        return "emt%d.%d.%s.%s.%s" % (start_index, round_index, chunk_index,
                                      self.dirpath.name, getpid())

    def make_job_name_identify(self, chunk_index):
        return "vit%d.%s.%s" % (chunk_index, self.dirpath.name, getpid())

    def make_gmtk_kwargs(self):
        num_tracks = self.num_tracks

        return dict(strFile=self.structure_filename,

                    of1=self.float_filelistpath,
                    fmt1="binary",
                    nf1=num_tracks,
                    ni1=0,
                    iswp1=False,

                    of2=self.int_filelistpath,
                    fmt2="binary",
                    nf2=0,
                    ni2=num_tracks,
                    iswp2=False,

                    verbosity=VERBOSITY)

    def make_chunk_mem_reqs(self):
        # XXX: should probably have different mem reqs for train or viterbi
        num_tracks = self.num_tracks

        chunk_lens = [end - start for chr, start, end in self.chunk_coords]

        self.chunk_lens = chunk_lens
        self.chunk_mem_reqs = [make_mem_req(chunk_len, num_tracks)
                               for chunk_len in chunk_lens]

    def queue_gmtk(self, session, prog, kwargs, job_name, mem_req,
                   native_specs={}):
        gmtk_cmdline = prog.build_cmdline(options=kwargs)

        # convoluted so I don't have to deal with a lot of escaping issues
        cmdline = BASH_CMDLINE + ['%s "$@"' % gmtk_cmdline[0]] + gmtk_cmdline

        if self.dry_run:
            print " ".join(gmtk_cmdline)
            return None

        job_tmpl = session.createJobTemplate()

        # shouldn't this be jobName? not in the Python DRMAA implementation
        # XXX: report upstream
        job_tmpl.name = job_name

        job_tmpl.remoteCommand = ENV_CMD
        job_tmpl.args = cmdline

        job_tmpl.outputPath = ":" + (self.output_dirpath / job_name)
        job_tmpl.errorPath = ":" + (self.error_dirpath / job_name)

        set_cwd_job_tmpl(job_tmpl)

        res_req = make_res_req(mem_req)
        job_tmpl.nativeSpecification = make_native_spec(l=res_req,
                                                        **native_specs)

        return session.runJob(job_tmpl)

    def queue_train(self, session, params_filename, start_index, round_index,
                    chunk_index, mem_req, hold_jid=None, **kwargs):
        kwargs["inputMasterFile"] = self.input_master_filename
        kwargs["inputTrainableParameters"] = params_filename
        kwargs["cppCommandOptions"] = make_cpp_options(params_filename)

        prog = self.train_prog
        name = self.make_job_name_train(start_index, round_index, chunk_index)
        native_specs = dict(hold_jid=hold_jid)

        return self.queue_gmtk(session, prog, kwargs, name, mem_req,
                               native_specs)

    def chunk_mem_reqs_decreasing(self):
        # sort chunks by decreasing size, so the most difficult chunks
        # are dropped in the queue first
        zipper = izip(self.chunk_lens, count(), self.chunk_mem_reqs)

        # XXX: use itertools instead of a generator
        for _, chunk_index, chunk_mem_req in sorted(zipper, reverse=True):
            yield chunk_index, chunk_mem_req

    def queue_train_parallel(self, session, params_filename, start_index,
                             round_index, **kwargs):
        queue_train_custom = partial(self.queue_train, session,
                                     params_filename, start_index, round_index)

        res = [] # task ids

        chunk_mem_reqs = list(self.chunk_mem_reqs_decreasing())
        last_chunk_index = chunk_mem_reqs[-1][0]
        for chunk_index, chunk_mem_req in chunk_mem_reqs:
            acc_filename = self.make_acc_filename(start_index, chunk_index)
            kwargs_chunk = dict(trrng=chunk_index, storeAccFile=acc_filename,
                                **kwargs)

            # -dirichletPriors T only on the last (smallest) chunk
            kwargs_chunk["dirichletPriors"] = (chunk_index == last_chunk_index)

            jobid = queue_train_custom(chunk_index, chunk_mem_req,
                                       **kwargs_chunk)
            res.append(jobid)

        return res

    def queue_train_bundle(self, session, parallel_jobids,
                           input_params_filename, output_params_filename,
                           start_index, round_index, **kwargs):
        """bundle step: take parallel accumulators and combine them
        """
        acc_filename = self.make_acc_filename(start_index,
                                              GMTK_INDEX_PLACEHOLDER)

        kwargs = \
            dict(outputTrainableParameters=output_params_filename,
                 trrng="nil",
                 loadAccRange="0:%s" % (self.num_chunks-1),
                 loadAccFile=acc_filename,
                 **kwargs)

        if self.dry_run:
            hold_jid = None
        else:
            hold_jid = ",".join(parallel_jobids)

        return self.queue_train(session, input_params_filename, start_index,
                                round_index, NAME_PLACEHOLDER, MEM_REQ_BUNDLE,
                                hold_jid, **kwargs)

    def run_triangulate(self):
        # XXX: should specify the triangulation file
        prog = self.prog_factory(TRIANGULATE_PROG)

        prog(strFile=self.structure_filename,
             verbosity=VERBOSITY)

    def run_train_start(self, session, start_index, interrupt_event):
        # make new files if you have more than one random start
        new = self.random_starts > 1

        self.save_input_master(start_index, new)
        self.set_params_filename(start_index, new)
        self.set_log_likelihood_filename(start_index, new)
        self.set_output_dirpaths(start_index)

        log_likelihood_filename = self.log_likelihood_filename
        last_log_likelihood = NINF
        log_likelihood = NINF
        round_index = 0

        stem_params_filename = self.params_filename
        last_params_filename = None
        curr_params_filename = None

        kwargs = self.train_kwargs

        queue_train_parallel = self.queue_train_parallel
        queue_train_bundle = self.queue_train_bundle

        timeout_no_wait = session.TIMEOUT_NO_WAIT

        while (round_index < MAX_EM_ITERS and
               is_training_progressing(last_log_likelihood, log_likelihood)):
            parallel_jobids = queue_train_parallel(session,
                                                   last_params_filename,
                                                   start_index, round_index,
                                                   **kwargs)

            curr_params_filename = extjoin(stem_params_filename,
                                           str(round_index))

            bundle_jobid = queue_train_bundle(session, parallel_jobids,
                                              last_params_filename,
                                              curr_params_filename,
                                              start_index, round_index,
                                              llStoreFile=\
                                                  log_likelihood_filename,
                                              **kwargs)

            last_params_filename = curr_params_filename

            if self.dry_run:
                log_likelihood = None
                break

            # wait for bundle to finish
            # XXXopt: polling in each thread is a bad way to do this
            # it would be best to use session.synchronize() centrally
            # and communicate to each thread when its job is done

            # the very best thing would be to eliminate the GIL lock
            # in the DRMAA wrapper
            job_info = None
            control = session.control
            terminate = session.TERMINATE
            while not job_info:
                try:
                    job_info = session.wait(bundle_jobid, timeout_no_wait)
                except ExitTimeoutError:
                    # ExitTimeoutError: not ready yet
                    interrupt_event.wait(SESSION_WAIT_TIMEOUT)
                except ValueError:
                    # ValueError: the job terminated abnormally
                    # so interrupt everybody
                    if self.keep_going:
                        return (None, None, None, None)
                    else:
                        interrupt_event.set()
                        raise

                # XXX: Py2.6+: use is_set() instead of isSet()
                if interrupt_event.isSet():
                    for jobid in parallel_jobids + [bundle_jobid]:
                        try:
                            print >>sys.stderr, "killing job %s" % jobid
                            control(jobid, terminate)
                        except BaseException, err:
                            print >>sys.stderr, ("ignoring exception: %r"
                                                 % err)
                    raise KeyboardInterrupt

            last_log_likelihood = log_likelihood
            log_likelihood = self.load_log_likelihood()

            print >>sys.stderr, "log likelihood = %s" % log_likelihood

            round_index += 1

        # log_likelihood and a list of src_filenames to save
        return (log_likelihood, self.input_master_filename,
                last_params_filename, log_likelihood_filename)

    def run_train(self):
        self.train_prog = self.prog_factory(EM_TRAIN_PROG)

        self.train_kwargs = dict(objsNotToTrain=self.dont_train_filename,
                                 maxEmIters=1,
                                 island=ISLAND,
                                 componentCache=COMPONENT_CACHE,
                                 lldp=LOG_LIKELIHOOD_DIFF_FRAC*100.0,
                                 **self.make_gmtk_kwargs())

        # save the destination file for input_master as we will be
        # generating new input masters for each start
        #
        # len(dst_filenames) == len(TRAIN_ATTRNAMES) == len(return value
        # of Runner.run_train_start())-1. This is asserted below.

        random_starts = self.random_starts
        assert random_starts >= 1

        # XXX: why did I have "if random_starts == 1:" preceding this line?
        self.save_input_master()

        if self.input_master_filename_new:
            input_master_filename = self.input_master_filename
        else:
            input_master_filename = None

        dst_filenames = [input_master_filename,
                         self.params_filename,
                         self.log_likelihood_filename]

        interrupt_event = Event()

        threads = []
        with Session() as session:
            try:
                for start_index in xrange(random_starts):
                    thread = RandomStartThread(self, session, start_index,
                                               interrupt_event)
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

        if not self.dry_run:
            # finds the max by log_likelihood
            src_filenames = max(start_params)[1:]

            assert (len(TRAIN_ATTRNAMES) == len(src_filenames)
                    == len(dst_filenames))

            zipper = zip(TRAIN_ATTRNAMES, src_filenames, dst_filenames)
            for name, src_filename, dst_filename in zipper:
                self.move_results(name, src_filename, dst_filename)

    def run_identify(self):
        if not self.input_master_filename:
            self.save_input_master()

        params_filename = self.params_filename

        prog = self.prog_factory(VITERBI_PROG)

        identify_kwargs = \
            dict(inputMasterFile=self.input_master_filename,
                 inputTrainableParameters=params_filename,

                 ofilelist=self.output_filelistname,
                 dumpNames=self.dumpnames_filename,

                 cppCommandOptions=make_cpp_options(params_filename),
                 **self.make_gmtk_kwargs())

        self.set_output_dirpaths("identify")

        # XXX: kill submitted jobs on exception
        jobids = []
        with Session() as session:
            queue_identify = partial(self.queue_gmtk, session, prog)

            for chunk_index, chunk_mem_req in self.chunk_mem_reqs_decreasing():
                identify_kwargs_chunk = dict(dcdrng=chunk_index,
                                             **identify_kwargs)
                job_name = self.make_job_name_identify(chunk_index)

                jobid = queue_identify(identify_kwargs_chunk, job_name,
                                       chunk_mem_req)
                jobids.append(jobid)

            session.synchronize(jobids, session.TIMEOUT_WAIT_FOREVER, True)

        if not self.dry_run:
            # XXXopt: could be done in parallel as well
            self.gmtk_out2bed()

    def _run(self):
        """
        main run, after dirname is specified
        """
        # XXXopt: use binary I/O to gmtk rather than ascii

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
    from ._util import OptionGroup

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

    with OptionGroup(parser, "Output files") as group:
        # XXX: separate this (now a single file) from output identity file
        # directory
        group.add_option("-b", "--bed", metavar="DIR",
                          help="use or create bed tracks in DIR",
                          default="out")

    with OptionGroup(parser, "Intermediate files") as group:
        # XXX: consider removing this option
        # this probably isn't necessary as observations are written
        # out pretty quickly now
        group.add_option("-o", "--observations", metavar="DIR",
                          help="use or create observations in DIR")

        group.add_option("-d", "--directory", metavar="DIR",
                          help="create all other files in DIR")


    with OptionGroup(parser, "Variables") as group:
        group.add_option("-r", "--random-starts", type=int,
                         default=RANDOM_STARTS, metavar="NUM",
                         help="randomize start parameters NUM times"
                         " (default 1)")

        group.add_option("--prior-strength", type=float, default=0,
                         metavar="RATIO",
                         help="use RATIO times the number of data counts as"
                         " the number of pseudocounts for the segment length"
                         " prior (default 0)")

    with OptionGroup(parser, "Flags") as group:
        group.add_option("-f", "--force", action="store_true",
                         help="delete any preexisting files")
        group.add_option("-I", "--no-identify", action="store_true",
                         help="do not identify segments")
        group.add_option("-T", "--no-train", action="store_true",
                         help="do not train model")
        group.add_option("-k", "--keep-going", action="store_true",
                         help="keep going in some threads even when you have"
                         " errors in another")
        group.add_option("-n", "--dry-run", action="store_true",
                         help="write all files, but do not run any"
                         " executables")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)
    dirname = options.directory
    bed_dirname = options.bed

    if dirname and not bed_dirname:
        bed_dirname = path(dirname) / BED_DIRNAME

    runner = Runner()

    runner.h5filenames = args
    runner.dirname = dirname
    runner.obs_dirname = options.observations
    runner.bed_dirname = bed_dirname
    runner.input_master_filename = options.input_master
    runner.structure_filename = options.structure
    runner.params_filename = options.trainable_params
    runner.include_coords_filename = options.include_coords

    runner.random_starts = options.random_starts
    runner.len_seg_strength = options.prior_strength
    runner.include_tracknames = options.track

    runner.delete_existing = options.force
    runner.train = not options.no_train
    runner.identify = not options.no_identify
    runner.dry_run = options.dry_run
    runner.keep_going = options.keep_going

    return runner()

if __name__ == "__main__":
    sys.exit(main())
