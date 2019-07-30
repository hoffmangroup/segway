#!/usr/bin/env python
from __future__ import absolute_import, division

__version__ = "$Revision$"

# Copyright 2009, 2011, 2012 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from math import ceil
from resource import getrlimit, RLIMIT_STACK
import sys

from .._util import MB
from .common import (CLEAN_SAFE_TIME, _JobTemplateFactory,
                     make_native_spec)

# qsub -w: switches off job validation
# qsub -j: switches off merging output and error
NATIVE_SPEC_DEFAULT = dict(w="n", j="n")

try:
    STACK_LIMIT = min(num for num in getrlimit(RLIMIT_STACK)
                      if num > 0)
except ValueError:
    STACK_LIMIT = 10*MB

class JobTemplateFactory(_JobTemplateFactory):
    def make_res_req(self, mem_usage, tmp_usage):
        # tmp_usage unused on SGE

        # By default the stack limit is so high that there is an
        # automatic crash when you also set a virtual memory limit and
        # then start new pthreads. Setting h_stack is necessary if you
        # are setting h_vmem.
        return [make_single_res_req("mem_requested", mem_usage),
                make_single_res_req("h_vmem", self.mem_limit),
                make_single_res_req("h_stack", STACK_LIMIT)]

    def make_native_spec(self):
        # qsub -l: resource requirement
        res_spec = make_native_spec(l=self.res_req)

        res = " ".join([self.native_spec,
                        make_native_spec(**NATIVE_SPEC_DEFAULT),
                        res_spec])

        return res

def make_single_res_req(name, mem):
    # round up to the next mebibyte
    return "%s=%dM" % (name, ceil(mem / MB))


def get_job_max_query_lifetime():
    """
    Get the maximum time in seconds a job's status can be queried (by DRMAA)
    """
    return CLEAN_SAFE_TIME


def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
