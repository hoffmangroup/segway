#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

## native specs work the same way as SGE
from .common import JobTemplateFactory as _JobTemplateFactory, _make_native_spec

def make_native_spec(args):
    return _make_native_spec(*args)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
