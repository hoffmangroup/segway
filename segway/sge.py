#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

from math import ceil
import sys

from optbuild import OptionBuilder_ShortOptWithSpace

# XXX: remove _sge_ from names

# N1 Grid Engine User's Guide chapter 3 page 71
SGE_MEM_SIZE_SUFFIXES = dict(K=2**10, M=2**20, G=2**30,
                             k=1e3, m=1e6, g=1e9)

QACCT_PROG = OptionBuilder_ShortOptWithSpace("qacct")

def parse_sge_qacct(text):
    res = {}

    for line in text.rstrip().split("\n"):
        if line.startswith("="):
            if res:
                yield res
            continue

        key, space, val = line.partition(" ")
        res[key] = val.strip()

    yield res

def convert_sge_mem_size(text):
    try:
        multiplier = SGE_MEM_SIZE_SUFFIXES[text[-1]]
        significand = float(text[:-1])

        res = significand * multiplier
    except KeyError:
        res = float(text)

    return int(ceil(res))

def fetch_sge_qacct_records(jobname):
    acct_text = QACCT_PROG.getoutput(j=jobname)

    return parse_sge_qacct(acct_text)

def fetch_sge_qacct_single_record(jobname):
    records = list(fetch_sge_qacct_records(jobname))
    assert len(records) == 1

    return records[0]

def get_sge_qacct_maxvmem(jobname):
    record = fetch_sge_qacct_single_record(jobname)
    return convert_sge_mem_size(record["maxvmem"])

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
