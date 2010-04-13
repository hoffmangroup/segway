#!/usr/bin/env python
from __future__ import division, with_statement

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

from drmaa import Session

def get_driver_name(session):
    drms_info = session.drmsInfo

    if drms_info.startswith("GE") or drms_info.startswith("SGE"):
        return "sge"
    elif drms_info.startswith("Platform LSF"):
        return "lsf"
    else:
        msg = ("unsupported distributed resource management system: %s"
               % drms_info)
        raise ValueError(msg)

# non-reentrant code
with Session() as _session:
    driver_name = get_driver_name(_session)

driver = __import__(driver_name, globals(), locals(), [driver_name], 1)
JobTemplateFactory = driver.JobTemplateFactory
make_native_spec = driver.make_native_spec

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
