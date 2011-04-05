#!/usr/bin/env python
from __future__ import division

"""compare_directory.py: compare two directories, using regexes

XXX: want to keep track of all files in new directory
"""

__version__ = "$Revision$"

## Copyright 2011 Michael M. Hoffman <mmh1@uw.edu>

from os import walk
from re import compile as re_compile, escape
import sys

from path import path

def get_dir_filenames(dirname):
    for dirbasename, dirnames, filenames in walk(dirname):
        relative_dirbasename = dirbasename.partition(dirname)[2]

        if relative_dirbasename.startswith("/"):
            relative_dirbasename = relative_dirbasename[1:]

        relative_dirpath = path(relative_dirbasename)

        try:
            dirnames.remove(".svn")
        except ValueError:
            pass

        for filename in filenames:
            yield str(relative_dirpath / filename)

# regular expression unescape
re_unescape = re_compile(r"\(%.*?%\)")
def make_regex(text):
    """
    make regex, escaping things that aren't with (% %)
    """
    spans = [match.span() for match in re_unescape.finditer(text)]

    res = ["^"]
    last_end = 0
    for start, end in spans:
        res.append(escape(text[last_end:start]))
        res.append(text[start+2:end-2]) # eliminate (% and %)
        last_end = end
    res.extend([escape(text[last_end:]), "$"])

    return re_compile("".join(res))

def compare_directory(template_dirname, query_dirname):
    query_filenames = set(get_dir_filenames(query_dirname))

    for template_filename in get_dir_filenames(template_dirname):
        re_template_filename = make_regex(template_filename)
        for query_filename in query_filenames:
            if re_template_filename.match(query_filename):
                query_filenames.remove(query_filename)
                break
        else:
            print >>sys.stderr, "query missing %s" % template_filename

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... TEMPLATEDIR QUERYDIR"
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) == 2:
        parser.error("incorrect number of arguments")

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return compare_directory(*args)

if __name__ == "__main__":
    sys.exit(main())
