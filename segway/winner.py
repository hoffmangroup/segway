#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, with_statement

"""winner.py: pick winning paramters when training run is cut short
"""

__version__ = "$Revision$"

## Copyright 2011, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from operator import itemgetter
from os import extsep
from re import compile as re_compile, escape
import sys

from path import Path

from ._util import (extjoin, EXT_MASTER, EXT_PARAMS, EXT_TAB, PREFIX_INPUT,
                    PREFIX_LIKELIHOOD, PREFIX_PARAMS, SUBDIRNAME_LOG,
                    SUBDIRNAME_PARAMS)

PAT_LIKELIHOOD = extjoin(PREFIX_LIKELIHOOD, "*", EXT_TAB)


def extjoin_escape(*args):
    return escape(extsep).join(args)

re_likelihood = re_compile(extjoin_escape(escape(PREFIX_LIKELIHOOD), "(.*)",
                                          escape(EXT_TAB)))


def get_likelihood_index(filepath):
    """
    returns a str
    """
    return re_likelihood.match(filepath.name).group(1)


def load_likelihood(filename):
    with open(filename) as infile:
        # get last line
        for line in infile:
            pass

    return float(line)


def enumerate_likelihoods(dirpath):
    log_dirpath = dirpath / SUBDIRNAME_LOG
    for filepath in log_dirpath.files(PAT_LIKELIHOOD):
        yield get_likelihood_index(filepath), load_likelihood(filepath)


def get_winning_instance(dirpath):
    return sorted(enumerate_likelihoods(dirpath), key=itemgetter(1, 0))[-1][0]


def enumerate_params_filenames(dirpath, instance):
    pattern = extjoin(PREFIX_PARAMS, instance, EXT_PARAMS, "*")

    for filename in dirpath.files(pattern):
        yield int(filename.rpartition(extsep)[2]), filename


def get_last_params_filename(dirpath, instance):
    return sorted(enumerate_params_filenames(dirpath, instance))[-1][-1]


def get_input_master_filename(dirpath, instance):
    return dirpath / extjoin(PREFIX_INPUT, instance, EXT_MASTER)


def print_and_copy(flag, getter, dirpath, instance, final_basename, copy,
                   clobber):
    if flag:
        srcpath = getter(dirpath, instance)
        print(srcpath)

        dstpath = dirpath / final_basename

        if clobber or not dstpath.exists():
            srcpath.copy2(dstpath)


def winner(dirname, params=True, input_master=True, copy=False, clobber=False):
    dirpath = Path(dirname)
    winning_instance = get_winning_instance(dirpath)

    params_dirpath = dirpath / SUBDIRNAME_PARAMS
    print_and_copy(input_master, get_input_master_filename, params_dirpath,
                   winning_instance, "input.master", copy, clobber)
    print_and_copy(params, get_last_params_filename, params_dirpath,
                   winning_instance, "params.params", copy, clobber)


def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... DIR"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    parser.add_option("-i", "--input-master", action="store_true",
                      help="print input master file name")
    parser.add_option("-p", "--params", action="store_true",
                      help="print parameters file name")
    parser.add_option("-c", "--copy", action="store_true",
                      help="copy files to final winning file locations")
    parser.add_option("--clobber", action="store_true",
                      help="overwrite existing files")

    options, args = parser.parse_args(args)

    if not len(args) == 1:
        parser.error("incorrect number of arguments")

    return options, args


def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    params = options.params
    input_master = options.input_master

    # if neither option is used, both are enabled
    if params is None and input_master is None:
        params = True
        input_master = True

    return winner(args[0], params=params, input_master=input_master,
                  copy=options.copy, clobber=options.clobber)

if __name__ == "__main__":
    sys.exit(main())
