#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

from math import ceil
import sys

from .._util import MB
from .common import _JobTemplateFactory, _make_native_spec

# qsub -w: switches off job validation
NATIVE_SPEC_DEFAULT = dict(w="n")

class JobTemplateFactory(_JobTemplateFactory):
    def make_res_req(self, mem_usage):
        return [make_single_res_req("mem_requested", mem_usage),
                make_single_res_req("h_vmem", self.mem_limit)]

    def make_native_spec(self):
        # qsub -l: resource requirement
        res_spec = make_native_spec(l=self.res_req)

        return " ".join([self.native_spec,
                         _make_native_spec(**NATIVE_SPEC_DEFAULT),
                         res_spec])

def make_single_res_req(name, mem):
    # round up to the next mebibyte
    return "%s=%dM" % (name, ceil(mem / MB))

def make_native_spec(args):
    return _make_native_spec(*args)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
