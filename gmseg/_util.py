#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@u.washington.edu>

from functools import partial
import shutil
import sys
from tempfile import mkdtemp

from pkg_resources import resource_filename, resource_string

DATA_PKG = "gmseg.data"

data_filename = partial(resource_filename, DATA_PKG)
data_string = partial(resource_string, DATA_PKG)

# NamedTemporaryFile is based somewhat on Python 2.5.2
# tempfile._TemporaryFileWrapper
#
# Original Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006 Python
# Software Foundation; All Rights Reserved
#
# License at http://www.python.org/download/releases/2.5.2/license/

# XXX: submit for inclusion in core

class NamedTemporaryDir(object):
    def __init__(self, *args, **kwargs):
        self.name = mkdtemp(*args, **kwargs)
        self.close_called = False
        self.rmtree = shutil.rmtree # want a function, not an unbound method

    def __enter__(self):
        return self

    def close(self):
        if not self.close_called:
            self.close_called = True
            self.rmtree(self.name)

    def __del__(self):
        self.close()

    def __exit__(self, exc, value, tb):
        self.close()

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
