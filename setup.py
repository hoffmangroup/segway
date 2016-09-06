#!/usr/bin/env python

"""segway: a way to segment the genome

Segway is a tool for easy pattern discovery and identification in
functional genomics data.
"""

from segway import __version__

# Copyright 2008-2014 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import sys
import subprocess

# required for OrderedDict
assert (2, 6) <= sys.version_info <= (3, 0)

from ez_setup import use_setuptools
use_setuptools()

from setuptools import find_packages, setup

MINIMUM_GMTK_VERSION = (1, 4, 2)
GMTK_VERSION_ERROR_MSG = """
GMTK version %s was detected.
Segway requires GMTK version %s or later to be installed.
Please update your GMTK version."""

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


def check_gmtk_version():
    """ Checks if the supported minimum GMTK version is installed """
    # Typical expected output from "gmtkPrint -version":
    # gmtkPrint (GMTK) 1.4.3
    # Mercurial id: 8995e40101d2 tip
    # checkin date: Fri Oct 30 10:51:44 2015 -0700

    # Older versions:
    # GMTK 1.4.0 (Mercurial id: bdf2718cc6ce tip checkin date: Thu Jun 25 12:31:56 2015 -0700)

    # Try to open gmtkPrint to get the version
    try:
        # blocks until finished
        output_string = subprocess.check_output(["gmtkPrint", "-version"])
    # If GMTK was not found
    except OSError:
        # Raise a runtime error stating that GMTK was not found on the path
        raise RuntimeError("GMTK cannot be found on your PATH.\nPlease "
                           "install GMTK from "
                           "http://melodi.ee.washington.edu/gmtk/ "
                           "before installing Segway.")

    output_lines = output_string.splitlines()

    # Check if there's only one line of output (for older versions)
    if len(output_lines) == 1:
        version_word_index = 1
    else:
        version_word_index = -1

    # Get the first line of output
    first_output_line = output_lines[0]

    # Get the version string from the proper word on the line
    current_version_string = first_output_line.split()[version_word_index]

    # Get the version number to compare with the minimum version
    current_version = map(int, current_version_string.split("."))
    version_zip = zip(current_version, MINIMUM_GMTK_VERSION)
    for current_version_number, minimum_version_number in version_zip:
        # If the version number (from most to least significant digit) is
        # ever less than the minimum
        if current_version_number < minimum_version_number:
            # Raise a runtime error stating the version found and the
            # minimum required
            minimum_version_string = ".".join(map(str, MINIMUM_GMTK_VERSION))
            raise RuntimeError(GMTK_VERSION_ERROR_MSG %
                               (current_version_string,
                                minimum_version_string))


def main():
    check_gmtk_version()
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

if __name__ == "__main__":
    main()
