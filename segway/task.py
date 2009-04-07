#!/usr/bin/env python
from __future__ import division

"""
task: wraps a GMTK subtask to reduce size of output
"""

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

from segway.run import VITERBI_PROG

def viterbi_task(chrom, start, end, *args):
    # a 2,000,000-frame output file is only 84 MiB so it is okay to
    # read the whole thing into memory

    output = VITERBI_PROG.getouput(*args)

def task(chrom, start, end, progname, *args):
    assert progname == VITERBI_PROG.prog

    start = int(start)
    end = int(end)

    viterbi_task(chrom, start, end, *args)

def main(args=sys.argv[1:]):
    return task(*args)

if __name__ == "__main__":
    sys.exit(main())
