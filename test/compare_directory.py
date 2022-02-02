#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

"""compare_directory.py: compare two directories, using regexes

XXX: want to keep track of all files in new directory
"""

## Copyright 2011-2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import filecmp
from os import walk
from re import compile as re_compile, escape
import sys
import tarfile

from path import Path
from six import viewitems, viewkeys
from six.moves import zip

from segway._util import maybe_gzip_open, SEGWAY_ENCODING
from segway.version import __version__

# marker for matching anywhere in file
ANY_LINE_MARKER = b"(%__ANY_LINE__%)"
ANY_LINE_MARKER_LENGTH = len(ANY_LINE_MARKER)

def get_dir_filenames(dirname):
    if (not Path(dirname).exists()):
        raise IOError("Directory %s not found" % dirname)
    for dirbasename, dirnames, filenames in walk(dirname):
        dirbasepath = Path(dirbasename)
        relative_dirbasename = dirbasename.partition(dirname)[2]

        if relative_dirbasename.startswith("/"):
            relative_dirbasename = relative_dirbasename[1:]

        relative_dirpath = Path(relative_dirbasename)

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
re_unescape = re_compile(b"\(%.*?%\)")

def make_regex(text):
    """
    make regex, escaping things that aren't with (% %)
    """
    spans = [match.span() for match in re_unescape.finditer(text)]

    res = [b"^"]
    last_end = 0
    for start, end in spans:
        res.append(escape(text[last_end:start]))
        res.append(text[start + 2:end - 2])  # eliminate (% and %)
        last_end = end
    res.extend([escape(text[last_end:]), b"$"])
    
    return re_compile(b"".join(res))


def compare_file(template_filename, query_filename):
    # quick comparison without regexes
    if filecmp.cmp(template_filename, query_filename, shallow=False):
        return True  # files are identical, skip slow regex stuff

    # If the files are different find and report the differences based on
    # embedded regex criteria
    
    # Create a template held out set for lines that can match elsewhere 
    match_anywhere_regexs = set()

    res = True
    with maybe_gzip_open(template_filename, mode = "rb") as template_file:
        with maybe_gzip_open(query_filename, mode = "rb") as query_file:
            for line_number, lines in enumerate(
                    zip(template_file, query_file),
                    start=1):
                template_line = lines[0]
                # If the template line can be matched anywhere
                if template_line.startswith(ANY_LINE_MARKER):
                    # Create regex from marker-removed template line
                    # Save to held out match anywhere line set
                    match_anywhere_regexs.add(
                        make_regex(template_line[ANY_LINE_MARKER_LENGTH:])
                    )
                else:
                    re_template = make_regex(template_line)
                    match = re_template.match(lines[1])
                    if not match:
                        res = False
                        print("Line %d differences for %s" % (
                            line_number,
                            template_filename))
                        print("Different line:\n%s" % (lines[1]))

    # If there are lines that may be matched anywhere
    if match_anywhere_regexs:
        # Fo each line in the the query file
        with maybe_gzip_open(query_filename, mode = "rb") as query_file:
            for query_line in query_file:
                # If a query line matches the regex in the match anywhere set
                regex_matches = []
                for template_regex in match_anywhere_regexs:
                    match = template_regex.match(query_line)
                    if match:
                        regex_matches.append(template_regex)

                # Remove any matched regexs from the match anywhere set
                match_anywhere_regexs.difference_update(regex_matches)
                # If there are no more match anywhere regexs
                if not match_anywhere_regexs:
                    # Stop searching
                    break

    # If there any match anywhere lines that do not match the query file
    if match_anywhere_regexs:
        # Report failure
        res = False
        for regex in match_anywhere_regexs:
            print("Cannot find any match for {}\n".format(regex.pattern))

    return res


class TestCounter(object):
    def __init__(self):
        self.num_error = 0
        self.num_success = 0

    def error(self, msg):
        self.num_error += 1
        print(" %s" % msg, file=sys.stderr)

    def success(self):
        self.num_success += 1


def compare_directory(template_dirname, query_dirname):
    counter = TestCounter()
    query_filenames = dict(get_dir_filenames(query_dirname))

    different_files_list = []

    template_filenames = get_dir_filenames(template_dirname)
    for template_filename_relative, template_filename in template_filenames:
        re_template_filename_relative = make_regex(template_filename_relative.encode(SEGWAY_ENCODING))

        query_filenames_items = viewitems(query_filenames)
        for query_filename_relative, query_filename in query_filenames_items:
            if re_template_filename_relative.match(query_filename_relative.encode(SEGWAY_ENCODING)):
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

    for query_filename_relative in viewkeys(query_filenames):
        counter.error("template directory missing %s"
                      % query_filename_relative)

    if counter.num_error == 0:
        msg = "PASS: %s and %s: %d files match" % (template_dirname,
                                                   query_dirname,
                                                   counter.num_success)
        print(msg, file=sys.stderr)
        return 0
    else:
        msg = "FAIL: %s and %s: %d files match;" \
              " %d files mismatch" % (template_dirname, query_dirname,
                                      counter.num_success, counter.num_error)

        base_directory_name = (Path(template_dirname)
                               .abspath().dirname().basename())
        tar_filename = base_directory_name + "-changes.tar.gz"

        with tarfile.open(tar_filename, "w:gz") as tar:
            print("Archiving in {}:".format(tar_filename))
            for template_filename, query_filename in different_files_list:
                tar.add(Path(template_filename).relpath())
                tar.add(Path(query_filename).relpath())
                print(template_filename, query_filename)

        print(msg, file=sys.stderr)
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
