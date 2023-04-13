#!/usr/bin/env python

"""gethelp.py: get help from a command-line script
"""

## Copyright 2011 Michael M. Hoffman <mmh1@uw.edu>

from importlib.metadata import entry_points
import sys

MODULE_NAME = 'segtools'

sys.path.insert(0, "..")

def gethelp(scriptname):
    console_scripts = entry_points()["console_scripts"]
    entry_point = [x for x in console_scripts if x.name==scriptname][0]
    entry = entry_point.value
    module_name, _, func_name = entry.partition(":")

    # __import__(module_name) usually returns the top-level package module only
    # so get our module out of sys.modules instead
    __import__(module_name)
    module = sys.modules[module_name]

    sys.argv[0] = scriptname
    getattr(module, func_name)(["--help"])

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... SCRIPTNAME"
    parser = OptionParser(usage=usage)

    options, args = parser.parse_args(args)

    if not len(args) == 1:
        parser.error("incorrect number of arguments")

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return gethelp(*args)

if __name__ == "__main__":
    sys.exit(main())
