#!/usr/bin/env python

"""segway: DESCRIPTION

LONG_DESCRIPTION
"""

__version__ = "0.1.11"

# Copyright 2008-2009 Michael M. Hoffman <mmh1@washington.edu>

import sys

# required for from __future__ import division, with_statement
assert sys.version_info >= (2, 5, 1)

from ez_setup import use_setuptools
use_setuptools()

from setuptools import find_packages, setup

doclines = __doc__.splitlines()
name, short_description = doclines[0].split(": ")
long_description = "\n".join(doclines[2:])

url = "http://noble.gs.washington.edu/~mmh1/software/%s/" % name.lower()
download_url = "%s%s-%s.tar.gz" % (url, name, __version__)

# XXX: remove these when the upstream packages are updated to fix these issues
dependency_links = ["http://pypi.python.org/packages/source/p/path.py/path-2.2.zip",
                    "http://gridengine.sunsource.net/files/documents/7/36/DRMAA-python-0.2.tar.gz"]

classifiers = ["Natural Language :: English",
               "Programming Language :: Python"]

# XXX: parallel should be an option, after the code is written
# correctly to fallback when it's not there
entry_points = """
[console_scripts]
segway = segway.run:main [parallel]
segway-res-usage = segway.res_usage:main [parallel]
segway-calc-distance = segway.calc_distance:main
segway-task = segway.task:main

h5histogram = segway.h5histogram:main
h5values = segway.h5values:main

gtf2bed = segway.gtf2bed:main
"""

# XXX: warn: make sure you have LDFLAGS unset if you are building numpy

# need optbuild>0.1.5 for Mixin_UseFullProgPath
# need tables>2.04 (>=r3761) because there is a CArray fill bug until then

install_requires = ["genomedata>0.1.0", "textinput", "optbuild>0.1.5",
                    "optplus", "tables>2.0.4", "numpy", "path", "colorbrewer"]

# XXX: ask if there is a way to specify this at the command-line
extras_require = dict(parallel=["DRMAA-python"])

if __name__ == "__main__":
    setup(name=name,
          version=__version__,
          description=short_description,
          author="Michael Hoffman",
          author_email="mmh1@washington.edu",
          url=url,
          download_url=download_url,
          classifiers=classifiers,
          long_description=long_description,
          dependency_links=dependency_links,
          install_requires=install_requires,
          extras_require=extras_require,
          zip_safe=False, # XXX: change back, this is just for better tracebacks

          # XXX: this should be based off of __file__ instead
          packages=find_packages("."),
          include_package_data=True,
          entry_points=entry_points
          )
