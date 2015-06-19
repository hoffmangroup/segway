#!/usr/bin/env python

"""segway: a way to segment the genome

Segway is a tool for easy pattern discovery and identification in
functional genomics data.
"""

__version__ = "1.2.2"

# Copyright 2008-2014 Michael M. Hoffman <michael.hoffman@utoronto.ca>

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
               "Development Status :: 5 - Production/Stable",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License v2 "
               "(GPLv2)",
               "Topic :: Scientific/Engineering :: Bio-Informatics",
               "Operating System :: Unix",
               "Programming Language :: Python",
               "Programming Language :: Python :: 2.6",
               "Programming Language :: Python :: 2.7"]

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
    # If the current version is a development version (modified or on an
    # untagged changeset)
    current_version = mgr.get_current_version()
    if current_version.endswith("dev"):
        # Return a custom version string based on changset id
        id_num = hg_id(mgr, "num")
        id_id = hg_id(mgr, "id")
        id_id = id_id.replace("+", "p")

        # "rel" always comes after r, so new numbering system is preserved
        # XXX: after 1.2.0 is released, we can change rel back to r
        return "%s.dev-rel%s-hg%s" % (__version__, id_num, id_id)
    # Otherwise return the current tagged version
    else:
        return current_version

if __name__ == "__main__":
    setup(name=name,
          version=__version__,
          description=short_description,
          author="Michael Hoffman",
          author_email="michael.hoffman@utoronto.ca",
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
