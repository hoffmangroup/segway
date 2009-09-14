#!/usr/bin/env python
from __future__ import division, with_statement

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

from drmaa import Session

# XXXXXXXX: monkey-patching, dirty hack to fix broken code

import drmaa.const

def status_to_string(status):
    return drmaa.const._JOB_PS[status]

drmaa.const.status_to_string = status_to_string

# XXXXXXXX: end monkey-patching

def get_cluster_key(session):
    drms_info = session.drmsInfo

    if drms_info.startswith("GE"):
        return "sge"
    elif drms_info == "LSF":
        return "lsf"
    else:
        msg = ("unsupported distributed resource management system: %s"
               % drms_info)
        raise ValueError(msg)

# non-reentrant code
with Session() as session:
    cluster_key = get_cluster_key(session)

driver = __import__(cluster_key)
JobTemplateFactory = driver.JobTemplateFactory
make_native_spec = driver.make_native_spec

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
