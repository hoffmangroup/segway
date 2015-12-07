#!/usr/bin/env python
from __future__ import division

"""compare_directory.py: compare two directories, using regexes

XXX: want to keep track of all files in new directory
"""

## Copyright 2011-2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import filecmp
from os import walk
from re import compile as re_compile, escape
import sys
import tarfile

from path import path

from segway._util import maybe_gzip_open
from segway.version import __version__

def get_dir_filenames(dirname):
    if (not path(dirname).exists()):
        raise IOError("Directory %s not found" % dirname)
    for dirbasename, dirnames, filenames in walk(dirname):
        dirbasepath = path(dirbasename)
        relative_dirbasename = dirbasename.partition(dirname)[2]

        if relative_dirbasename.startswith("/"):
            relative_dirbasename = relative_dirbasename[1:]

        relative_dirpath = path(relative_dirbasename)

        try:
            dirnames.remove(".svn")
        except ValueError:
            pass

        dirnames.sort()

        for filename in sorted(filenames):
            filename_relative = str(relative_dirpath / filename)
            # not really absolute, but more so than relative
            filename_absolute = str(dirbasepath / filename)
            yield filename_relative, filename_absolute

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
        res.append(text[start + 2:end - 2])  # eliminate (% and %)
        last_end = end
    res.extend([escape(text[last_end:]), "$"])

    return re_compile("".join(res))


def compare_file(template_filename, query_filename):
    # quick comparison without regexes
    if filecmp.cmp(template_filename, query_filename, shallow=False):
        return True  # files are identical, skip slow regex stuff

    # If the files are different find and report the differences based on
    # embedded regex criteria
    res = True
    with maybe_gzip_open(template_filename) as template_file:
        with maybe_gzip_open(query_filename) as query_file:
            for line_number, lines in enumerate(
                    zip(template_file, query_file),
                    start=1):
                re_template = make_regex(lines[0])
                match = re_template.match(lines[1])
                if not match:
                    res = False
                    print "Line %d differences for %s" % (
                        line_number,
                        template_filename)
                    print "Different line:\n%s" % (lines[1])

    return res


class TestCounter(object):
    def __init__(self):
        self.num_error = 0
        self.num_success = 0

    def error(self, msg):
        self.num_error += 1
        print >>sys.stderr, " %s" % msg

    def success(self):
        self.num_success += 1


def compare_directory(template_dirname, query_dirname):
    counter = TestCounter()
    query_filenames = dict(get_dir_filenames(query_dirname))

    different_files_list = []

    template_filenames = get_dir_filenames(template_dirname)
    for template_filename_relative, template_filename in template_filenames:
        re_template_filename_relative = make_regex(template_filename_relative)

        query_filenames_items = query_filenames.iteritems()
        for query_filename_relative, query_filename in query_filenames_items:
            if re_template_filename_relative.match(query_filename_relative):
                del query_filenames[query_filename_relative]
                if compare_file(template_filename, query_filename):
                    counter.success()
                else:
                    counter.error("diff '%s' '%s'" % (template_filename,
                                                      query_filename))
                    different_files_list.append((template_filename,
                                                 query_filename))

                break
        else:
            counter.error("query directory missing %s" % template_filename)

    for query_filename_relative in query_filenames.iterkeys():
        counter.error("template directory missing %s"
                      % query_filename_relative)

    if counter.num_error == 0:
        msg = "PASS: %s and %s: %d files match" % (template_dirname,
                                                   query_dirname,
                                                   counter.num_success)
        print >>sys.stderr, msg
        return 0
    else:
        msg = "FAIL: %s and %s: %d files match;" \
              " %d files mismatch" % (template_dirname, query_dirname,
                                      counter.num_success, counter.num_error)

        base_directory_name = (path(template_dirname)
                               .abspath().dirname().basename())
        tar_filename = base_directory_name + "-changes.tar.gz"

        with tarfile.open(tar_filename, "w:gz") as tar:
            print "Archiving in {}:".format(tar_filename)
            for template_filename, query_filename in different_files_list:
                tar.add(template_filename)
                tar.add(query_filename)
                print template_filename, query_filename

        print >>sys.stderr, msg
        return 1


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
