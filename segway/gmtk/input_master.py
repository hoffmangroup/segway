#!/usr/bin/env python
from collections import OrderedDict

from numpy import array, ndarray

def array2text(a):
    ndim = a.ndim
    if ndim == 1:
        return " ".join(map(str, a))
    else:
        delimiter = "\n" * (ndim - 1)
        return delimiter.join(array2text(row) for row in a)

class InputMaster(list):
    """
    Master class which contains a list of sections a is able to write them
    out to file properly.
    """
    def __str__(self):
        return "\n".join([str(item) for item in self])

    def append(self, item):
        if not isinstance(item, Section):
            raise ValueError("Only section objects may be saved to an InputMaster")
        list.append(self, item)


class Section(OrderedDict):
    """
    Defines a section, which contains gmtk objects of all the same type and
    supports writing them to file. This is a subclass of OrderedDict, where the
    keys are set to represent the object names, while their values will be a
    subclass of object.
    """
    def __str__(self):
        section_objects = []
        section_objects.append(str(len(self)))
        for index, object_name in enumerate(self):
            section_objects = section_objects + \
                list(map(str, [index, object_name, self[object_name]]))

        return "\n".join(section_objects)

    def kind(self):
        section_kind = None
        # Ensure all members of the section are the same kind
        for obj in self.values():
            if not section_kind:
                section_kind = obj.kind
            else:
                assert section_kind == obj.kind
        return section_kind


class InlineSection(Section):
    def __str__(self):
        header = "{}_IN_FILE infile\n\n".format(self.kind())
        return "".join([header, Section.__str__(self)])

class FileSection(Section):
    def __init__(self, filename, items):
        raise NotImplementedError("Reading from file is not yet supported")


class Object(str):
    def __new__(cls, content, kind):
        return str.__new__(cls, content)

    def __init__(self, content, kind):
        self.kind = kind


class Array(ndarray):
    def __new__(cls, *args, **kwargs):
        return array(*args, **kwargs).view(cls)


class DenseCPT(Array):
    def __array_finalize__(self, obj):
        self.kind = "DENSE_CPT"

    def __str__(self):
        # Find the number of parents, their cardinalities and this cardinality
        num_parents = str(len(self.shape)-1)
        shape_list = list(map(str, self.shape))
        shape_list.insert(0, num_parents)
        shape_line = " ".join(shape_list)
        return "\n".join([shape_line, array2text(self.__array__())])
