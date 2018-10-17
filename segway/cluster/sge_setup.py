#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, with_statement

"""
sge_setup: setup mem_requested on each node
"""

__version__ = "$Revision$"

# Copyright 2010, 2012, 2013 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import sys
from tempfile import NamedTemporaryFile

from optbuild import OptionBuilder_ShortOptWithSpace
from six import viewitems

QCONF_PROG = OptionBuilder_ShortOptWithSpace("qconf")
QSTAT_PROG = OptionBuilder_ShortOptWithSpace("qstat")

MEM_TOTAL_STR = "hl:mem_total="
HOSTNAME_STR = "qf:hostname="

# XXX: only works if there is no user named "fakeuser" on the system
# fix by adding random suffix and checking
USERNAME = "fakeuser"

# XXX: this may be fragile
OUTPUT_RECORD_SEPARATOR = \
    "----------------------------------------------------------------------------\n"

def add_complex_mem_requested():
    """add the mem_requested complex to GE configuration
    """
    tempfile = NamedTemporaryFile("w", suffix=".txt", prefix="qconf.",
                                  delete=False)

    prior_complex_text = QCONF_PROG.getoutput(sc=True)
    print(prior_complex_text, file=tempfile)
    print("mem_requested\tmr\tMEMORY\t<=\tYES\tYES\t0\t10", file=tempfile)

    tempfile.close()

    # XXX: would be better if this were robust to mem_requested already existing
    QCONF_PROG(Mc=tempfile.name)

def get_mem_totals():
    stat_texts_text = QSTAT_PROG.getoutput(F="hostname,mem_total", u=USERNAME)
    stat_texts = stat_texts_text.split(OUTPUT_RECORD_SEPARATOR)

    res = {}

    # skip header
    for stat_text in stat_texts[1:]:
        lines_dict = {}

        lines = stat_text.splitlines()
        for line in lines:
            key, equals, value = line.strip().partition("=")
            if equals == "=":
                lines_dict[key] = value

        # XXX: if hl:mem_total not specified, print a warning instead
        # of failing

        # when you have multiple queues, qstat might report values for
        # the hostname once for each queue. we'll overwrite instead of
        # checking for equality
        res[lines_dict["qf:hostname"]] = lines_dict["hl:mem_total"]

    return res

def modify_complex_values_mem_requested():
    mem_totals = get_mem_totals()

    for hostname, mem_total in viewitems(mem_totals):
        QCONF_PROG("-mattr", "exechost", "complex_values",
                   "mem_requested=%s" % mem_total, hostname)

def sge_setup():
    add_complex_mem_requested()
    modify_complex_values_mem_requested()

    # extra newline at beginning to space from qconf messages
    print("""
===========================================================================
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

It's also a good idea to verify the configuration by running:

qhost -F mem_requested
===========================================================================
""", file=sys.stderr)

# XXX: a dry-run option would be good

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
