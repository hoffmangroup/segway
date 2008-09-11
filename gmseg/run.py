#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@u.washington.edu>

from itertools import izip
from math import floor, log10
from os import close, extsep, fdopen
import shutil
from string import Template
from struct import calcsize, unpack
import sys
from tempfile import mkstemp

from optbuild import OptionBuilder_ShortOptWithSpace

# XXX: change to relative import after entry points defined
from gmseg._util import data_filename, data_string, NamedTemporaryDir
from gmseg.bed import read

# XXX: should be options
NUM_SEGS = 2
MAX_EM_ITERS = 100
TEMPDIR_PREFIX = "gmseg-"
VERBOSITY = 0

# XXX: should be an option, including a temporary file
OUTPUT_FILENAME = "output.out"

TRIANGULATE_PROG = OptionBuilder_ShortOptWithSpace("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_ShortOptWithSpace("gmtkEMtrainNew")
VITERBI_PROG = OptionBuilder_ShortOptWithSpace("gmtkViterbiNew")

MEAN_TMPL = "$index mean_${seg}_${obs} 1 0.0"
COVAR_TMPL = "$index covar_${seg}_${obs} 1 1.0"
MC_TMPL = "$index 1 0 mc_${seg}_${obs} " \
    "mean_${seg}_${obs} covar_${seg}_${obs}"
MX_TMPL = "$index 1 mx_${seg}_${obs} 1 dpmfOne mc_${seg}_${obs}"

NAME_COLLECTION_TMPL = "$obs_index collection_seg_${obs} 2"
NAME_COLLECTION_CONTENTS_TMPL = "mx_${seg}_${obs}"

WIG_EXT = "wig"

TRACK_FMT = "browser position %s:%s-%s"

# XXX: this could be a dict instead
WIG_HEADER = 'track type=wiggle_0 name=gmseg ' \
    'description="segmentation by gmseg" visibility=full viewLimits=0:1' \
    'autoScale=off'

def save_include():
    return data_filename("seg.inc")

def save_template(resource, mapping, tempdirname):
    tmpl = Template(data_string(resource))

    text = tmpl.substitute(mapping)

    resource_part = resource.rpartition(".tmpl")
    stem = resource_part[0] or resource_part[2]
    stem_part = stem.rpartition(".")
    prefix = stem_part[0] + "."
    suffix = "." + stem_part[2]

    temp_fd, temp_filename = mkstemp(suffix, prefix, tempdirname)

    with fdopen(temp_fd, "w+") as temp_file:
        temp_file.write(text)

    return temp_filename

def save_structure(include_filename, num_obs, tempdirname):
    observation_tmpl = Template(data_string("observation.tmpl"))
    observation_substitute = observation_tmpl.substitute

    observations = \
        "\n".join(observation_substitute(observation_index=observation_index)
                  for observation_index in xrange(num_obs))

    mapping = dict(include_filename=include_filename,
                   observations=observations)

    return save_template("seg.str.tmpl", mapping, tempdirname)

#def save_output_master():
#    return data_filename("output.master")

def make_spec(name, items):
    items[:0] = ["%s_IN_FILE inline" % name, str(len(items)), ""]

    return "\n".join(items) + "\n"

def make_dt_spec(num_obs):
    return make_spec("DT", ["%d seg_obs%d BINARY_DT" % (index, index)
                            for index in xrange(num_obs)])

def make_items_multiseg(tmpl, num_segs, num_obs):
    substitute = Template(tmpl).substitute

    res = []

    for seg_index in xrange(num_segs):
        seg = "seg%d" % seg_index
        for obs_index in xrange(num_obs):
            obs = "obs%d" % obs_index
            mapping = dict(seg=seg, obs=obs,
                           seg_index=seg_index, obs_index=obs_index,
                           index=num_obs*seg_index + obs_index)

            res.append(substitute(mapping))

    return res

def make_spec_multiseg(name, *args, **kwargs):
    return make_spec(name, make_items_multiseg(*args, **kwargs))

def make_mean_spec(num_obs):
    return make_spec_multiseg("MEAN", MEAN_TMPL, NUM_SEGS, num_obs)

def make_covar_spec(num_obs):
    return make_spec_multiseg("COVAR", COVAR_TMPL, NUM_SEGS, num_obs)

def make_mc_spec(num_obs):
    return make_spec_multiseg("MC", MC_TMPL, NUM_SEGS, num_obs)

def make_mx_spec(num_obs):
    return make_spec_multiseg("MX", MX_TMPL, NUM_SEGS, num_obs)

def make_name_collection_spec(num_obs):
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

def save_input_master(include_filename, num_obs, tempdirname):
    mapping = dict(include_filename=include_filename,
                   dt_spec=make_dt_spec(num_obs),
                   mean_spec=make_mean_spec(num_obs),
                   covar_spec=make_covar_spec(num_obs),
                   mc_spec=make_mc_spec(num_obs),
                   mx_spec=make_mx_spec(num_obs),
                   name_collection_spec=make_name_collection_spec(num_obs))

    return save_template("input.master.tmpl", mapping, tempdirname)

def load_observations_bed(bed_filename):
    with open(bed_filename) as bed_file:

        # XXXopt: this copy/paste is necessary because otherwise the
        # bed_file will be closed before the iterator values are received
        for datum in read(bed_file):
            yield datum

    # returns an iterator
    # each iteration yields a .bed.Datum

def load_observations_list(bed_filelistname):
    with open(bed_filelistname) as bed_filelist:
        for line in bed_filelist:
            bed_filename = line.rstrip()

            yield load_observations_bed(bed_filename)

    # returns an iterator
    # each iteration yields an iterator
    # each iteration yields a .bed.Datum

def load_observations_lists(bed_filelistnames):
    res = []

    observations_lists = [load_observations_list(bed_filelistname)
                          for bed_filelistname in bed_filelistnames]

    # each iteration yields an iterator
    observations_lists_zipped = izip(*observations_lists)

    for observations_bed_iterators in observations_lists_zipped:
        res.append(zip(*observations_bed_iterators))

    # returns a list
    # each item is a list
    # each subitem is a tuple
    # each sub-subitem is a .bed.Datum
    return res

def save_observations_gmtk(observation_rows, prefix, tempdirname):
    temp_fd, temp_filename = mkstemp(".obs", prefix, tempdirname)

    with fdopen(temp_fd, "w+") as temp_file:
        for observation_row in observation_rows:
            print >>temp_file, " ".join(datum.score
                                        for datum in observation_row)

    return temp_filename

def save_observations_list(observation_rows_list, tempdirname):
    prefix_tmpl = "obs" + make_prefix_tmpl(len(observation_rows_list))

    temp_fd, temp_filename = mkstemp(".list", dir=tempdirname)

    with fdopen(temp_fd, "w+") as temp_file:
        for index, observation_rows in enumerate(observation_rows_list):
            prefix = prefix_tmpl % index
            text = save_observations_gmtk(observation_rows, prefix,
                                          tempdirname)

            print >>temp_file, text

    return temp_filename

def mkstemp_closed(*args, **kwargs):
    temp_fd, temp_filename = mkstemp(*args, **kwargs)
    close(temp_fd)

    return temp_filename

def make_prefix_tmpl(num_filenames):
    # make sure there aresufficient leading zeros
    return "%%0%dd." % (int(floor(log10(num_filenames))) + 1)

def save_output_filelist(num_filenames, tempdirname):
    temp_fd, temp_filename = mkstemp(".list", "output", tempdirname)

    prefix_tmpl = "out" + make_prefix_tmpl(num_filenames)
    output_filenames = \
        [mkstemp_closed(".out", prefix_tmpl % index, tempdirname)
         for index in xrange(num_filenames)]

    with fdopen(temp_fd, "w+") as temp_file:
        for output_filename in output_filenames:
            print >>temp_file, output_filename

    return output_filenames, temp_filename

def save_dumpnames(num_segs, tempdirname):
    temp_fd, temp_filename = mkstemp(".list", "dumpnames", tempdirname)

    with fdopen(temp_fd, "w+") as temp_file:
        print >>temp_file, "seg"

    return temp_filename

def read_gmtk_out(infile):
    data = infile.read()

    fmt = "%dL" % (len(data) / calcsize("L"))
    return unpack(fmt, data)

def write_wig(outfile, output, observations):
    first_datum = observations[0][0]
    last_datum = observations[-1][0]

    chrom = first_datum.chrom
    start = first_datum.chromStart
    end = last_datum.chromEnd

    print >>outfile, TRACK_FMT % (chrom, start, end)
    print >>outfile, WIG_HEADER

    for score, data in zip(output, observations):
        # this assumes that the relevant values for all items in data
        # are the same
        #
        # XXX: check this
        datum = data[0]

        row = [datum.chrom, datum.chromStart, datum.chromEnd, str(score)]
        print >>outfile, "\t".join(row)

def load_gmtk_out_save_wig(filename, observations):
    wigfilename = extsep.join([filename.rpartition(".")[0], WIG_EXT])

    with open(filename) as gmtkfile:
        with open(wigfilename, "w") as wigfile:
            return write_wig(wigfile, read_gmtk_out(gmtkfile), observations)

def gmtk_out2wig(filenames, observations_list):
    for filename, observations in zip(filenames, observations_list):
        load_gmtk_out_save_wig(filename, observations)

def run_triangulate(structure_filename):
    TRIANGULATE_PROG(strFile=structure_filename,
                     verbosity=VERBOSITY)
    # XXX: creates trifile that needs to be destroyed
    # XXX: best to use a temporary directory for everything--see poly source

def run_em_train(structure_filename, input_master_filename,
                 output_filename, gmtk_filelistname, num_obs):

    EM_TRAIN_PROG(strFile=structure_filename,

                  inputMasterFile=input_master_filename,
                  outputTrainableParameters=output_filename,

                  of1=gmtk_filelistname,
                  fmt1="ascii",
                  nf1=num_obs,
                  ni1=0,

                  maxEmIters=MAX_EM_ITERS,
                  verbosity=VERBOSITY)

def run_viterbi(structure_filename, input_master_filename, output_filename,
                gmtk_filelistname, num_obs, output_filelistname,
                dumpnames_filename):
    # XXX: change to tmpdir first to remove jtinfo litter

    VITERBI_PROG(strFile=structure_filename,

                 inputMasterFile=input_master_filename,
                 inputTrainableParameters=output_filename,

                 ofilelist=output_filelistname,
                 dumpNames=dumpnames_filename,

                 of1=gmtk_filelistname,
                 fmt1="ascii",
                 nf1=num_obs,
                 ni1=0,

                 cppCommandOptions="-DVITERBI",
                 verbosity=VERBOSITY)

def run(*bed_filelistnames):
    # XXX: use binary I/O to gmtk rather than ascii
    # XXX: register atexit for cleanup_resources

    # XXX: allow specification of directory instead of tmp
    with NamedTemporaryDir(prefix=TEMPDIR_PREFIX) as tempdir:
        tempdirname = tempdir.name

        num_obs = len(bed_filelistnames)

        include_filename = save_include()
        structure_filename = save_structure(include_filename, num_obs,
                                            tempdirname)
        run_triangulate(structure_filename)

        input_master_filename = save_input_master(include_filename, num_obs,
                                                  tempdirname)

        # input: multiple lists -> multiple filenames -> one column
        # output: one list -> multiple_filenames -> multiple columns
        observations_list = load_observations_lists(bed_filelistnames)

        # XXX: assert that the appropriate coordinates for each Datum
        # are aligned
        gmtk_filelistname = save_observations_list(observations_list,
                                                   tempdirname)

        # XXX: add option to skip EM training
        run_em_train(structure_filename, input_master_filename,
                     OUTPUT_FILENAME, gmtk_filelistname, num_obs)

        output_filenames, output_filelistname = \
            save_output_filelist(len(observations_list), tempdirname)
        dumpnames_filename = save_dumpnames(NUM_SEGS, tempdirname)

        run_viterbi(structure_filename, input_master_filename, OUTPUT_FILENAME,
                    gmtk_filelistname, num_obs, output_filelistname,
                    dumpnames_filename)

        gmtk_out2wig(output_filenames, observations_list)

        import pdb; pdb.set_trace()

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILELIST..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return run(*args)

if __name__ == "__main__":
    sys.exit(main())
