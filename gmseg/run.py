#!/usr/bin/env python
from __future__ import division, with_statement

"""
run: DESCRIPTION
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@u.washington.edu>

from itertools import izip
from os import remove
from string import Template
import sys
from tempfile import NamedTemporaryFile

from optbuild import OptionBuilder_ShortOptWithSpace

# XXX: change to relative import after entry points defined
from gmseg._util import data_filename, data_string
from gmseg.bed import read

# XXX: should be an option, including a temporary file
OUTPUT_FILENAME = "output.out"

TRIANGULATE_PROG = OptionBuilder_ShortOptWithSpace("gmtkTriangulate")
EM_TRAIN_PROG = OptionBuilder_ShortOptWithSpace("gmtkEMtrainNew")
VITERBI_PROG = OptionBuilder_ShortOptWithSpace("gmtkViterbiNew")

def save_include():
    return data_filename("seg.inc")

def save_structure(include_filename, num_observations):
    observation_tmpl = Template(data_string("observation.tmpl"))
    observation_substitute = observation_tmpl.substitute

    observations = \
        "\n".join(observation_substitute(observation_index=observation_index)
                  for observation_index in xrange(num_observations))

    structure_tmpl = Template(data_string("seg.str.tmpl"))
    mapping = dict(include_filename=include_filename,
                   observations=observations)
    structure_str = structure_tmpl.substitute(mapping)

    structure_file = NamedTemporaryFile(suffix=".str")
    structure_file.write(structure_str)
    structure_file.flush()

    return structure_file

#def save_output_master():
#    return data_filename("output.master")

def save_input_master():
    return data_filename("input.master")

def load_observations_bed(bed_filename):
    with open(bed_filename) as bed_file:
        for datum in read(bed_file):
            yield datum.score

    # returns an iterator
    # each iteration yields a score

def load_observations_list(bed_filelistname):
    with open(bed_filelistname) as bed_filelist:
        for line in bed_filelist:
            bed_filename = line.rstrip()

            yield load_observations_bed(bed_filename)

    # returns an iterator
    # each iteration yields an iterator
    # each iteration yields a score

def load_observations_lists(bed_filelistnames):
    res = []

    observations_lists = [load_observations_list(bed_filelistname)
                          for bed_filelistname in bed_filelistnames]

    # each iteration yields an iterator
    observations_lists_zipped = izip(*observations_lists)

    for observations_bed_iterators in observations_lists_zipped:
        res.append(zip(*observations_bed_iterators))

    return res

def save_observations_gmtk(observation_rows):
    res = NamedTemporaryFile(suffix=".obs")

    for observation_row in observation_rows:
        print >>res, " ".join(observation_row)

    res.flush()

    return res

def save_observations_list(observation_rows_list):
    temp_files = []

    for observation_rows in observation_rows_list:
        temp_files.append(save_observations_gmtk(observation_rows))

    observations_list_file = NamedTemporaryFile(suffix=".list")
    for temp_file in temp_files:
        print >>observations_list_file, temp_file.name

    observations_list_file.flush()

    temp_files.append(observations_list_file)

    # keep temp_files to keep files around
    # they are unlinked when the object is deleted
    return observations_list_file.name, temp_files

def run_triangulate(structure_filename):
    TRIANGULATE_PROG(strFile=structure_filename)
    # XXX: creates trifile that needs to be destroyed
    # XXX: best to use a temporary directory for everything--see poly source

def run_em_train(structure_filename, input_master_filename,
                 output_filename, gmtk_filelistname, num_observations):

    EM_TRAIN_PROG(strFile=structure_filename,

                  inputMasterFile=input_master_filename, XXX update with new stuff
                  outputTrainableParameters=output_filename,

                  of1=gmtk_filelistname,
                  fmt1="ascii",
                  nf1=num_observations,
                  ni1=0,

                  maxEmIters=5)

def run_viterbi():
    pass # XXX

def run(*bed_filelistnames):
    # XXX register atexit for cleanup_resources

    num_observations = len(bed_filelistnames)

    include_filename = save_include()
    structure_file = save_structure(include_filename, num_observations)
    structure_filename = structure_file.name
    run_triangulate(structure_filename)

    input_master_filename = save_input_master()
#    output_master_filename = save_output_master()

    # input: multiple lists -> multiple filenames -> one column
    # output: one list -> multiple_filenames -> multiple columns
    observations_list = load_observations_lists(bed_filelistnames)

    gmtk_filelistname, temp_files = save_observations_list(observations_list)

    run_em_train(structure_filename, input_master_filename, OUTPUT_FILENAME,
                 gmtk_filelistname, num_observations)
    import pdb; pdb.set_trace()

    run_viterbi()

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
