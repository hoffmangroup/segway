#!/usr/bin/env python

"""segway: a way to segment the genome

Segway is a tool for easy pattern discovery and identification in
functional genomics data.
"""

__version__ = "1.2.0"

# Copyright 2008-2014 Michael M. Hoffman <mhoffman@uhnresearch.ca>

import sys

# required for OrderedDict
assert (2, 6) <= sys.version_info <= (3, 0)

from ez_setup import use_setuptools
use_setuptools()

from setuptools import find_packages, setup

doclines = __doc__.splitlines()
name, short_description = doclines[0].split(": ")
long_description = "\n".join(doclines[2:])

url = "http://pmgenomics.ca/hoffmanlab/proj/%s/" % name.lower()
download_url = "%ssrc/%s-%s.tar.gz" % (url, name, __version__)

classifiers = ["Natural Language :: English",
               "Programming Language :: Python"]

entry_points = """
[console_scripts]
segway = segway.run:main
segway-task = segway.task:main
segway-layer = segway.layer:main
segway-winner = segway.winner:main
"""

# XXX: warn: make sure you have LDFLAGS unset if you are building numpy

# need optbuild>0.1.11 for OptionBuilder_ShortOptWithEquals
# need tables>2.04 (>=r3761) because there is a CArray fill bug until then
# genomedata>1.3.1 for Chromosome.__getitem__[..., array] support

install_requires = ["genomedata>1.3.1", "textinput", "optbuild>0.1.10",
                    "optplus>0.1.0", "tables>2.0.4", "numpy", "forked-path",
                    "colorbrewer", "drmaa>=0.4a3"]

def hg_id(mgr, kind):
    return mgr._invoke("id", "--%s" % kind).strip()

def calc_version(mgr, options):
    id_num = hg_id(mgr, "num")
    id_id = hg_id(mgr, "id")
    id_id = id_id.replace("+", "p")

    # "rel" always comes after r, so new numbering system is preserved
    # XXX: after 1.2.0 is released, we can change rel back to r
    return "%s.dev-rel%s-hg%s" % (__version__, id_num, id_id)

if __name__ == "__main__":
    setup(name=name,
          version=__version__,
          description=short_description,
          author="Michael Hoffman",
          author_email="mhoffman@uhnresearch.ca",
          url=url,
          download_url=download_url,
          classifiers=classifiers,
          long_description=long_description,
          install_requires=install_requires,
          setup_requires=["hgtools"],
          zip_safe=False, # XXX: change back, this is just for better tracebacks

          # XXX: this should be based off of __file__ instead
          packages=find_packages("."),
          include_package_data=True,
          use_vcs_version={"version_handler": calc_version},
          entry_points=entry_points
          )
