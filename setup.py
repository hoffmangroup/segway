#!/usr/bin/env python

"""segway: DESCRIPTION

LONG_DESCRIPTION
"""

__version__ = "1.0.0"

# Copyright 2008-2011 Michael M. Hoffman <mmh1@washington.edu>

import sys

# required for from __future__ import division, with_statement
assert sys.version_info >= (2, 5, 1)

from ez_setup import use_setuptools
use_setuptools()

from setuptools import find_packages, setup

doclines = __doc__.splitlines()
name, short_description = doclines[0].split(": ")
long_description = "\n".join(doclines[2:])

url = "http://noble.gs.washington.edu/proj/%s/" % name.lower()
download_url = "%s%s-%s.tar.gz" % (url, name, __version__)

classifiers = ["Natural Language :: English",
               "Programming Language :: Python"]

# XXX: parallel should be an option, after the code is written
# correctly to fallback when it's not there
entry_points = """
[console_scripts]
segway = segway.run:main [parallel]
segway-calc-distance = segway.calc_distance:main
segway-task = segway.task:main
segway-layer = segway.layer:main

h5histogram = segway.h5histogram:main
h5values = segway.h5values:main

gtf2bed = segway.gtf2bed:main
"""

# XXX: warn: make sure you have LDFLAGS unset if you are building numpy

# need optbuild>0.1.5 for Mixin_UseFullProgPath
# need tables>2.04 (>=r3761) because there is a CArray fill bug until then

install_requires = ["genomedata>0.1.5", "textinput", "optbuild>=0.1.10",
                    "optplus>0.1.0", "tables>2.0.4", "numpy", "forked-path",
                    "colorbrewer"]

# XXX: ask if there is a way to specify this at the command-line
extras_require = dict(parallel=["drmaa>=0.4a3"])

if __name__ == "__main__":
    setup(name=name,
          version=__version__,
          description=short_description,
          author="Michael Hoffman",
          author_email="mmh1@uw.edu",
          url=url,
          download_url=download_url,
          classifiers=classifiers,
          long_description=long_description,
          install_requires=install_requires,
          extras_require=extras_require,
          zip_safe=False, # XXX: change back, this is just for better tracebacks

          # XXX: this should be based off of __file__ instead
          packages=find_packages("."),
          include_package_data=True,
          entry_points=entry_points
          )
