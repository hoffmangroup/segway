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

from ._util import (extjoin, EXT_MASTER, EXT_PARAMS, EXT_TAB,
                    PREFIX_INPUT, PREFIX_LIKELIHOOD, PREFIX_PARAMS,
                    SUBDIRNAME_LOG, SUBDIRNAME_PARAMS)

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

def load_likelihood(filename, validate = False):
    best_line = -1e+20
    with open(filename) as infile:
        # if validate find best line, else return last 
        for index, line in enumerate(infile):
            if float(line) > best_line and validate:
                best_line = float(line)
                best_index = index
    if validate:
        return best_index, best_line
    return [str(index), float(line)]

def enumerate_likelihoods(dirpath, validate):
    log_dirpath = dirpath / SUBDIRNAME_LOG
    if not log_dirpath.files(PAT_LIKELIHOOD):
        yield [None] + load_likelihood(log_dirpath / extjoin(PREFIX_LIKELIHOOD, EXT_TAB), validate)
    for filepath in log_dirpath.files(PAT_LIKELIHOOD):
        yield [get_likelihood_index(filepath)] + load_likelihood(filepath, validate)


def get_winning_instance(dirpath, validate):
    return sorted(enumerate_likelihoods(dirpath, validate), key=itemgetter(2, 0, 1))[-1][0:-1]


def enumerate_params_filenames(dirpath, instance):
    pattern = extjoin(PREFIX_PARAMS, instance, EXT_PARAMS, "*")

    for filename in dirpath.files(pattern):
        yield int(filename.rpartition(extsep)[2]), filename


def get_last_params_filename(dirpath, instance):
    if not instance[0]:
        instance[0] = "0"
    return dirpath / extjoin(PREFIX_PARAMS, instance[0], EXT_PARAMS, instance[1])


def get_input_master_filename(dirpath, instance):
    return dirpath / extjoin(PREFIX_INPUT, instance, EXT_MASTER)

def get_log_likelihood_filename(dirpath, instance):
    return dirpath / extjoin(PREFIX_LIKELIHOOD, instance, EXT_TAB)


def print_and_copy(flag, getter, dirpath, instance, final_basename, clobber):
    if flag:
        srcpath = getter(dirpath, instance)
        print(srcpath)

        dstpath = dirpath / final_basename

        if clobber or not dstpath.exists():
            srcpath.copy2(dstpath)


def winner(dirname, params=True, input_master=True, likelihood = False, 
           clobber=False, validate = False):
    dirpath = Path(dirname)
    winning_instance = get_winning_instance(dirpath, validate)

    params_dirpath = dirpath / SUBDIRNAME_PARAMS
    log_dirpath = dirpath / SUBDIRNAME_LOG
    print_and_copy(input_master, get_input_master_filename, params_dirpath,
                   winning_instance[0], "input.master", clobber)
    print_and_copy(params, get_last_params_filename, params_dirpath,
                   winning_instance, "params.params", clobber)
    print_and_copy(likelihood, get_log_likelihood_filename, log_dirpath,
                   winning_instance[0], "likelihood.tab", clobber)

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
