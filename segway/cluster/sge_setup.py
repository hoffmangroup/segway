#!/usr/bin/env python
from __future__ import division, with_statement

"""
sge_setup: setup mem_requested on each node
"""

__version__ = "$Revision$"

# Copyright 2010 Michael M. Hoffman <mmh1@uw.edu>

import sys
from tempfile import NamedTemporaryFile

from optbuild import OptionBuilder_ShortOptWithSpace
# from path import path

QCONF_PROG = OptionBuilder_ShortOptWithSpace("qconf")
QSTAT_PROG = OptionBuilder_ShortOptWithSpace("qstat")

MEM_TOTAL_STR = "hl:mem_total="
HOSTNAME_STR = "qf:hostname="

USERNAME = "fakeuser"

OUTPUT_RECORD_SEPARATOR = \
    "----------------------------------------------------------------------------\n"

def sge_setup():
    tempfile = NamedTemporaryFile("w", suffix=".txt", prefix="qconf.")

    print >>tempfile, "mem_requested\tmr\tMEMORY\t<=\tYES\tYES\t0\t10"
    # XXX: after Python 2.6, use tempfile.close() with NamedTemporaryFile(delete=False)
    tempfile.flush()

    QCONF_PROG(Mc=tempfile.name)

    # path(tempfile.name).unlink()

    stat_texts_text = QSTAT_PROG.getoutput(F="hostname,mem_total", u=USERNAME)
    stat_texts = stat_texts_text.split(OUTPUT_RECORD_SEPARATOR)

    mem_totals = {}

    # skip header
    for stat_text in stat_texts[1:]:
        lines_dict = {}

        lines = stat_text.splitlines()
        for line in lines:
            key, equals, value = line.strip().partition("=")
            if equals == "=":
                lines_dict[key] = value

        # you might get a hostname reported multiple times for each queue
        # we'll overwrite instead of checking for equality
        mem_totals[lines_dict["qf:hostname"]] = lines_dict["hl:mem_total"]

    for hostname, mem_total in mem_totals.iteritems():
        QCONF_PROG("-mattr", "exechost", "complex_values",
                   "mem_requested=%s" % mem_total, hostname)

    print >>sys.stderr, """\
The mem_requested resource is now configured on your cluster. While
Segway uses this resource to control its own jobs, we recommend
setting a default allocation for all other jobs by editing the default
SGE request. Otherwise they will not request any memory unless
explicitly specified, which can lead to conflicts with
mem_requested-aware applications. For example, you can edit the file
$SGE_ROOT/$SGE_CELL/common/sge_request to add this line:

-l mem_requested=2G

This will request 2 GiB of RAM for an otherwise unspecfied job, which
is usually a good idea on a cluster where hosts have, say, 8 cores and
16 GiB of RAM.
"""

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)

    options, args = parser.parse_args(args)

    if not len(args) == 0:
        parser.error("incorrect number of arguments")

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return sge_setup(*args)

if __name__ == "__main__":
    sys.exit(main())
