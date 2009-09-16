#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

from math import ceil
import sys

## native specs work the same way as SGE
from .._util import KB, MB
from .common import _JobTemplateFactory, make_native_spec

class JobTemplateFactory(_JobTemplateFactory):
    def make_res_req(self, mem_usage):
        return "rusage[mem=%s]" % ceil(mem_usage / MB)

    def make_native_spec(self):
        mem_limit_spec = ceil(self.mem_limit / KB)

        # bsub -R: resource requirement
        # bsub -v: hard virtual memory limit
        res_spec = make_native_spec(R=self.res_req, v=mem_limit_spec)

        return " ".join([self.native_spec, res_spec])

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
