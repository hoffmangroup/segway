#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@u.washington.edu>

from functools import partial
import sys

from pkg_resources import resource_filename, resource_string

DATA_PKG = "gmseg.data"

data_filename = partial(resource_filename, DATA_PKG)
data_string = partial(resource_string, DATA_PKG)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
