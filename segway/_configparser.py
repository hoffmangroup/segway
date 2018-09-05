#!/usr/bin/env python
from __future__ import absolute_import, division, with_statement

__version__ = "$Revision$"

# Copyright 2009 Michael M. Hoffman <michael.hoffman@utoronto.ca>

import sys

from six import string_types
from six.moves import StringIO
from six.moves.configparser import RawConfigParser

CONFIGPARSER_SECTION = "all"

class OneSectionRawConfigParser(RawConfigParser):
    """
    for UNIX configuration files which lake a section header
    """
    def read(self, filenames):
        if not isinstance(filenames, string_types):
            raise NotImplementedError

        with open(filenames) as infile:
            text = infile.read()

        buffer = StringIO("[%s]\n%s" % (CONFIGPARSER_SECTION, text))

        RawConfigParser.readfp(self, buffer)

        assert self.sections() == [CONFIGPARSER_SECTION]

    def readfp(self, *args, **kwargs):
        raise NotImplementedError

    def options(self):
        return RawConfigParser.options(self, CONFIGPARSER_SECTION)

    def get(self, option):
        return RawConfigParser.get(self, CONFIGPARSER_SECTION, option)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
