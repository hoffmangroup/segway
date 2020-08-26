"""segway: a way to segment the genome

Segway is a tool for easy pattern discovery and identification in
functional genomics data.
"""

# Copyright 2008-2014 Michael M. Hoffman <michael.hoffman@utoronto.ca>

from __future__ import absolute_import

import sys

if (sys.version_info[0] == 2 and sys.version_info[1] < 7) or \
   (sys.version_info[0] == 3 and sys.version_info[1] < 4):
    print("Segway requires Python version 2.7 or 3.4 or later")
    sys.exit(1)

from setuptools import find_packages, setup

doclines = __doc__.splitlines()
name, short_description = doclines[0].split(": ")
long_description = "\n".join(doclines[2:])

url = "https://%s.hoffmanlab.org" % name.lower()
download_url = "https://pypi.python.org/pypi/%s" % name.lower()

classifiers = ["Natural Language :: English",
               "Development Status :: 5 - Production/Stable",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU General Public License v2 "
               "(GPLv2)",
               "Topic :: Scientific/Engineering :: Bio-Informatics",
               "Operating System :: Unix",
               "Programming Language :: Python",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: 3"]

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
# genomedata>=1.4.2 for both Python 2 and 3 support
# optplus>=0.2 for both Python 2 and 3 support

setup_requires = ["setuptools_scm"] # source control management packaging
install_requires = ["genomedata>=1.4.2", "autolog>=0.2.0",
                    "textinput>=0.2.0", "optbuild>=0.2.0",
                    "optplus>=0.2.0", "tables>2.0.4", "numpy", "path.py>=11",
                    "colorbrewer>=0.2.0", "drmaa>=0.4a3", "six"]


def main():
    setup(name=name,
          use_scm_version=True,
          description=short_description,
          author="Michael Hoffman",
          author_email="michael.hoffman@utoronto.ca",
          url=url,
          download_url=download_url,
          classifiers=classifiers,
          long_description=long_description,
          setup_requires=setup_requires,
          install_requires=install_requires,
          zip_safe=False, # XXX: change back, this is just for better tracebacks
          packages=find_packages("."),
          include_package_data=True,
          entry_points=entry_points
          )

if __name__ == "__main__":
    main()
