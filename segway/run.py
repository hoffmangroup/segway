#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from cStringIO import StringIO
from contextlib import closing
from errno import EEXIST, ENOENT
from itertools import count, izip
from math import floor, log10
from os import extsep, makedirs
from random import random
from shutil import move, rmtree
from string import Template
from struct import calcsize, unpack
import sys

from numpy import amin, amax, array, finfo, float32, isnan
from numpy.random import uniform
from optbuild import OptionBuilder_ShortOptWithSpace_TF
from path import path
from tables import openFile

from ._util import (data_filename, data_string, init_num_obs,
                    NamedTemporaryDir, PKG, walk_continuous_supercontigs)

# XXX: should be options
NUM_SEGS = 2
MAX_EM_ITERS = 100
VERBOSITY = 30
TEMPDIR_PREFIX = PKG + "-"
COVAR_TIED = True # would need to expand to MC, MX to change
MAX_CHUNKS = 1000
ISLAND = False
WIG_DIRNAME = "out"

# for extra memory savings, set to (False) or (not ISLAND)
COMPONENT_CACHE = True

# number of frames in a segment must be at least number of frames in model
MIN_FRAMES = 2
MAX_FRAMES = 10000000 # 10 million

# defaults
RANDOM_STARTS = 1

# replace NAN with SENTINEL to avoid warnings
MIN_FLOAT = finfo(float32).min
SENTINEL = MIN_FLOAT

# programs
TRIANGULATE_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkEMtrainNew")
VITERBI_PROG = OptionBuilder_ShortOptWithSpace_TF("gmtkViterbiNew")

# extensions and suffixes
EXT_WIG = "wig"
EXT_LIST = "list"
EXT_OBS = "obs"
EXT_OUT = "out"

PREFIX_LIST = "features"
PREFIX_CHUNK = "chunk"

SUFFIX_LIST = extsep + EXT_LIST
SUFFIX_OUT = extsep + EXT_OUT
SUFFIX_OBS = extsep + EXT_OBS

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

MC_TMPL = "$index 1 0 mc_${seg}_${obs} mean_${seg}_${obs} covar_${obs}"
MX_TMPL = "$index 1 mx_${seg}_${obs} 1 dpmf_always mc_${seg}_${obs}"

NAME_COLLECTION_TMPL = "$obs_index collection_seg_${obs} 2"
NAME_COLLECTION_CONTENTS_TMPL = "mx_${seg}_${obs}"

TRACK_FMT = "browser position %s:%s-%s"
FIXEDSTEP_FMT = "fixedStep chrom=%s start=%s step=1 span=1"

# XXX: this could be specified as a dict instead
WIG_HEADER = 'track type=wiggle_0 name=%s ' \
    'description="Segmentation by %s" visibility=dense viewLimits=0:1 ' \
    'autoScale=off' % (PKG, PKG)

TRAIN_ATTRNAMES = ["input_master_filename", "trainable_params_filename"]

def extjoin(*args):
    return extsep.join(args)

def save_template(filename, resource, mapping, dirname=None,
                  delete_existing=False):
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

        filename = path(dirname) / extjoin(prefix, ext)

    with open(filename, "w+") as outfile:
        tmpl = Template(data_string(resource))
        text = tmpl.substitute(mapping)

        outfile.write(text)

    return filename

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

# XXX: numpy
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
    return array([uniform(low, high)
                  for seg_index in xrange(num_segs)])

def make_mean_spec(num_segs, num_obs, mins, maxs):
    rands = make_rands(mins, maxs, num_segs)

    return make_spec_multiseg("MEAN", MEAN_TMPL, num_segs, num_obs, rands)

def make_covar_spec(num_segs, num_obs, mins, maxs):
    if COVAR_TIED:
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

    fmt = "%dL" % (len(data) / calcsize("L"))
    return unpack(fmt, data)

def write_wig(outfile, output, (chrom, start, end)):
    print >>outfile, TRACK_FMT % (chrom, start, end)
    print >>outfile, WIG_HEADER
    print >>outfile, FIXEDSTEP_FMT % (chrom, start)

    print >>outfile, "\n".join(map(str, output))

def load_gmtk_out_save_wig(chunk_coord, gmtk_outfilename, wig_filename):
    with open(gmtk_outfilename) as gmtk_outfile:
        data = read_gmtk_out(gmtk_outfile)

        with open(wig_filename, "w") as wig_file:
            return write_wig(wig_file, data, chunk_coord)

class Runner(object):
    def __init__(self, **kwargs):
        # filenames
        self.h5filenames = None
        self.feature_filelistpath = None

        self.include_filename = None
        self.input_master_filename = None
        self.structure_filename = None

        self.trainable_params_filename = None
        self.dirname = None
        self.is_dirname_temp = False
        self.log_likelihood_filename = None
        self.dont_train_filename = None

        self.dumpnames_filename = None
        self.output_filelistname = None
        self.output_filenames = None

        self.obs_dirname = None
        self.wig_dirname = None

        # data
        self.num_chunks = None
        self.chunk_coords = None
        self.mins = None
        self.maxs = None

        # variables
        self.num_segs = NUM_SEGS
        self.random_starts = RANDOM_STARTS

        # flags
        self.delete_existing = False
        self.triangulate = True
        self.train = True # EM train # this should become an int for num_starts
        self.identify = True # viterbi
        self.dry_run = False

        self.__dict__.update(kwargs)

    def load_log_likelihood(self):
        with open(self.log_likelihood_filename) as infile:
            return float(infile.read().strip())

    def set_trainable_params_filename(self, new=False):
        # if this is not run and trainable_params_filename is
        # unspecified, then it won't be passed to gmtkViterbiNew

        trainable_params_filename = self.trainable_params_filename
        if not new and trainable_params_filename:
            if (not self.delete_existing
                and path(trainable_params_filename).exists()):
                # it already exists and you don't want to force regen
                self.train = False
        else:
            filenamebase = extjoin("params", "params")
            self.trainable_params_filename = self.dirpath / filenamebase

    def set_log_likelihood_filename(self):
        filenamebase = extjoin("likelihood", "ll")
        self.log_likelihood_filename = self.dirpath / filenamebase

    def make_dir(self, dirname):
        if self.delete_existing:
            # just always try to delete it
            try:
                rmtree(dirname)
            except OSError, err:
                if err.errno != ENOENT:
                    raise

        return makedirs(dirname)

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
            dirpath = path(self.dirname)

            orig_filepath.copy(dirpath)
            return dirpath / orig_filepath.name

    def save_include(self):
        self.include_filename = self.save_resource(RES_INC)

    def save_structure(self):
        observation_tmpl = Template(data_string("observation.tmpl"))
        observation_sub = observation_tmpl.substitute

        num_obs = self.num_obs
        observations = \
            "\n".join(observation_sub(obs_index=obs_index,
                                      nonmissing_index=num_obs+obs_index)
                      for obs_index in xrange(num_obs))

        mapping = dict(include_filename=self.include_filename,
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

        for h5filename in self.h5filenames:
            print >>sys.stderr, h5filename
            chrom = path(h5filename).namebase

            with openFile(h5filename) as chromosome:
                chromosome_attrs = chromosome.root._v_attrs

                try:
                    chromosome_mins = chromosome_attrs.mins
                    chromosome_maxs = chromosome_attrs.maxs
                except AttributeError:
                    # this means there is no data for that chromosome
                    continue

                if mins is None:
                    mins = chromosome_mins
                    maxs = chromosome_maxs
                else:
                    mins = amin([chromosome_mins, mins], 0)
                    maxs = amax([chromosome_maxs, maxs], 0)

                for supercontig, continuous in \
                        walk_continuous_supercontigs(chromosome):
                    num_obs = init_num_obs(num_obs, continuous)

                    supercontig_attrs = supercontig._v_attrs
                    supercontig_start = supercontig_attrs.start

                    starts = supercontig_attrs.chunk_starts
                    ends = supercontig_attrs.chunk_ends

                    ## iterate through chunks and write
                    for start, end in zip(starts, ends):
                        num_frames = end - start
                        if not MIN_FRAMES <= num_frames <= MAX_FRAMES:
                            text = " skipping segment of length %d" \
                                % num_frames
                            print >>sys.stderr, text
                            continue

                        chunk_start = supercontig_start + start
                        chunk_end = supercontig_start + end
                        chunk_coords.append((chrom, chunk_start, chunk_end))

                        chunk_filepath = make_obs_filepath(chunk_index)

                        print >>feature_filelist, chunk_filepath
                        print >>sys.stderr, " " + chunk_filepath

                        if not chunk_filepath.exists():
                            rows = continuous[start:end, ...]
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

    def save_input_master(self, new=False):
        num_segs = self.num_segs
        num_obs = self.num_obs
        mins = self.mins
        maxs = self.maxs

        include_filename = self.include_filename

        if new:
            input_master_filename = None
        else:
            input_master_filename = self.input_master_filename

        dense_cpt_spec = make_dense_cpt_spec(num_segs)
        mean_spec = make_mean_spec(num_segs, num_obs, mins, maxs)
        covar_spec = make_covar_spec(num_segs, num_obs, mins, maxs)
        mc_spec = make_mc_spec(num_segs, num_obs)
        mx_spec = make_mx_spec(num_segs, num_obs)
        name_collection_spec = make_name_collection_spec(num_segs, num_obs)

        self.input_master_filename = \
            save_template(input_master_filename, RES_INPUT_MASTER_TMPL,
                          locals(), self.dirname, self.delete_existing)
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
        self.make_obs_dir()
        self.save_observations() # do first, because it sets self.num_obs

        self.save_include()
        self.save_structure()

        if self.train:
            self.save_dont_train()
            self.set_trainable_params_filename() # might turn off self.train
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
        wig_filebasename_fmt = "".join([PKG, prefix_fmt, EXT_WIG])

        wig_dirpath = path(self.wig_dirname)
        wig_filepath_fmt = wig_dirpath / wig_filebasename_fmt

        # chunk_coord = (chrom, chromStart, chromEnd)
        zipper = izip(count(), self.output_filenames, self.chunk_coords)
        for index, gmtk_outfilename, chunk_coord in zipper:
            wig_filename = wig_filepath_fmt % index

            load_gmtk_out_save_wig(chunk_coord, gmtk_outfilename,
                                   wig_filename)

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

    def run_triangulate(self):
        # XXX: should specify the triangulation file
        prog = self.prog_factory(TRIANGULATE_PROG)

        prog(strFile=self.structure_filename,
                         verbosity=VERBOSITY)

    def run_train(self):
        prog = self.prog_factory(EM_TRAIN_PROG)

        # XXX: this can be a parallelization point
        kwargs = dict(strFile=self.structure_filename,
                      objsNotToTrain=self.dont_train_filename,
                      llStoreFile=self.log_likelihood_filename,

                      of1=self.feature_filelistpath,
                      fmt1="ascii",
                      nf1=self.num_obs,
                      ni1=self.num_obs,

                      maxEmIters=MAX_EM_ITERS,
                      verbosity=VERBOSITY,
                      island=ISLAND,
                      componentCache=COMPONENT_CACHE)

        dst_filenames = [self.input_master_filename,
                         self.trainable_params_filename]

        # list of tuples(log_likelihood, input_master_filename,
        #                trainable_params_filename)
        start_params = []

        for start_index in xrange(self.random_starts):
            # XXX: re-add the ability to set your own starting parameters,
            # with new=start_index
            # (copy from existing rather than using it on command-line)
            self.save_input_master(new=True)
            self.set_trainable_params_filename(new=True)

            input_master_filename = self.input_master_filename
            trainable_params_filename = self.trainable_params_filename

            prog(inputMasterFile=input_master_filename,
                 outputTrainableParameters=trainable_params_filename,
                 **kwargs)

            if not self.dry_run:
                start_params.append((self.load_log_likelihood(),
                                     input_master_filename,
                                     trainable_params_filename))

        if not self.dry_run:
            src_filenames = max(start_params)[1:]

            zipper = zip(TRAIN_ATTRNAMES, src_filenames, dst_filenames)
            for name, src_filename, dst_filename in zipper:
                self.move_results(name, src_filename, dst_filename)

    def run_identify(self):
        if not self.input_master_filename:
            self.save_input_master()

        trainable_params_filename = self.trainable_params_filename
        if trainable_params_filename:
            cpp_options = "-DUSE_TRAINABLE_PARAMS"
        else:
            cpp_options = None

        prog = self.prog_factory(VITERBI_PROG)

        prog(strFile=self.structure_filename,

             inputMasterFile=self.input_master_filename,
             inputTrainableParameters=trainable_params_filename,

             ofilelist=self.output_filelistname,
             dumpNames=self.dumpnames_filename,

             of1=self.feature_filelistpath,
             fmt1="ascii",
             nf1=self.num_obs,
             ni1=self.num_obs,

             cppCommandOptions=cpp_options,
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

    parser.add_option("--trainable-params", "-t", metavar="FILE",
                      help="use or create trainable parameters in FILE")

    parser.add_option("--directory", "-d", metavar="DIR",
                      help="create all other files in DIR")

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
    runner.trainable_params_filename = options.trainable_params

    runner.random_starts = options.random_starts

    runner.delete_existing = options.force
    runner.train = not options.no_train
    runner.identify = not options.no_identify
    runner.dry_run = options.dry_run

    return runner()

if __name__ == "__main__":
    sys.exit(main())
