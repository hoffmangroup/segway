#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# common stuff
# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

from optbuild import Mixin_NoConvertUnderscore, OptionBuilder_ShortOptWithSpace

NATIVE_SPEC_PROG = (Mixin_NoConvertUnderscore
                    + OptionBuilder_ShortOptWithSpace)() # do not run

# guard space to prevent going over mem_requested allocation
MEM_GUARD = 0

class JobError(RuntimeError):
    pass

class _JobTemplateFactory(object):
    set_template_output_error = True

    def __init__(self, template, mem_usage_progression, output_filename,
                 error_filename):
        self.native_spec = template.nativeSpecification
        self.mem_usage_progression = mem_usage_progression

        # set here so that it can be overridden for LSF
        self.output_filename = output_filename
        self.error_filename = error_filename

        if self.set_template_output_error:
            template.outputPath = ":" + output_filename
            template.errorPath = ":" + error_filename

        self.template = template

    def __call__(self, trial_index):
        res = self.template

        try:
            mem_usage = self.mem_usage_progression[trial_index]
        except IndexError:
            print >>sys.stderr, "Edge of memory usage progression reached without success."
            print >>sys.stderr, "Check for errors in output/e subdirectory."
            print >>sys.stderr, "See the troubleshooting section of the Segway documentation."
            raise RuntimeError("Edge of memory usage progression reached without success.")

        self.mem_limit = calc_mem_limit(mem_usage)
        self.res_req = self.make_res_req(mem_usage)
        res.nativeSpecification = self.make_native_spec()

        return res

    def make_res_req(self, mem_usage):
        # pure virtual function
        raise NotImplementedError

    def make_native_spec(self):
        # pure virtual function
        raise NotImplementedError

def make_native_spec(*args, **kwargs):
    return " ".join(NATIVE_SPEC_PROG.build_args(args=args, options=kwargs))

def calc_mem_limit(mem_usage):
    """
    amount of memory allowed to be used (less than that requested)
    """
    return int(mem_usage - MEM_GUARD)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
