from __future__ import division, print_function
import sys 
from collections import OrderedDict
import numpy as np
from numpy import array, ndarray

COMPONENT_TYPE_DIAG_GAUSSIAN = 0


def array2text(a):
    """
    Convert multi-dimensional array to text.
    :param a: array
    :return:
    """
    ndim = a.ndim
    if ndim == 1:
        return " ".join(map(str, a))
    else:
        delimiter = "\n" * (ndim - 1)
        return delimiter.join(array2text(row) for row in a)

class Object(str):
    def __new__(cls, _name, content, _kind):
        return str.__new__(cls, content)

    def __init__(self, name, content, kind):
        self.kind = kind
        self.name = name

class Array(ndarray):
    def __new__(cls, *args):
        """
        :param input_array: ndarray
        :return:
        """
        input_array = array(args)
        obj = np.asarray(input_array).view(cls)
        return obj


class Section(OrderedDict):
    """
    Contains GMTK objects of a single type and supports writing them to file.
    Key: name of GMTK object
    Value: GMTK object
    """
    def __init__(self):
        """
        Initialize an empty Section object.
        """
        super(Section, self).__init__()
        
    def __getattr__(self, name):
        if not name.startswith('_'):
            return self[name]
        OrderedDict.__getattr__(self, name)

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            if not self.kind() == value.kind:
                raise ValueError("Object has incorrect type.")
            self[name] = value
        else:
            OrderedDict.__setattr__(self, name, value)
        
    def kind(self):
        """
        Return string attribute kind of all GMTK objects in this Section object.
        :return: str: type of all GMTK objects in this Section object
        """
        section_kind = None
        for obj in self.values():
            if not section_kind:
                section_kind = obj.kind
            else:
                assert section_kind == obj.kind, "Objects must be of same type."
        return section_kind

class InlineSection(Section):

    def __str__(self):
        """
        Returns inline string representation of this Section object by calling
        the individual GMTK object's __str__().
        :return:
        """
        # if no gmtk objects
        if len(self) == 0:
            return ""

        lines = ["{}_IN_FILE inline".format(self.kind())]
        lines.append(str(len(self)) + "\n")  # total number of gmtk objects
        for i in range(len(self)):
            lines.append(str(i))  # index of gmtk object
            lines.append(list(self)[i])  # name of gmtk object
            lines.append(list(self.values())[i].__str__())
            # string representation of gmtk object

        return "\n".join(lines)


class InlineMCSection(InlineSection):
    """
    Special InlineSection subclass which contains MC objects.
    Attributes:
        mean: InlineSection object which point to InputMaster.mean
        covar: InlineSection object which point to InputMaster.covar
    """
    def __init__(self, mean, covar):
        """
        :param mean: InlineSection: InlineSection object which point to
        InputMaster.mean
        :param covar: InlineSection: InlineSection object which point to
        InputMaster.covar
        """
        super(InlineMCSection, self).__init__()
        self.mean = mean
        self.covar = covar
        
    def __setattr__(self, key, value):
        OrderedDict.__setattr__(self, key, value)


    def __str__(self):
        """
        Returns string representation of all MC objects contained in this
        InlineMCSection by calling the individual MC object's __str__().
        :return:
        """
        if len(self) == 0:
            return ""
        else:
            lines = ["{}_IN_FILE inline".format(self.kind())]
            lines.append(str(len(self)) + "\n")  # total number of MC objects
            for i in range(len(self)):
                lines.append(str(i))  # index of MC object
                # check if dimension of Mean and Covar of this MC are the same
                obj = list(self.values())[i]
                mean_name = obj.mean
                covar_name = obj.covar
                if not self.mean[mean_name].get_dimension() == self.covar[covar_name].get_dimension():
                    # TODO delete MC? redefine?
                    raise ValueError("Inconsistent dimensions of mean and covar associated to MC.")
                else:
                    lines.append(str(self.mean[mean_name].get_dimension()))
                    # dimension of MC
                    lines.append(str(obj.component_type))  # component type
                    lines.append(list(self)[i])  # name of MC
                    lines.append(obj.__str__())  # string representation of MC obj

            lines.append("\n")
            return "\n".join(lines)


class InlineMXSection(InlineSection):
    """
        Special InlineSection subclass which contains MX objects.
        Attributes:
            dpmf: InlineSection object which point to InputMaster.dpmf
            components: InlineSection object which point to InputMaster.mc
        """

    def __init__(self, dpmf, mc):
        """
        :param dpmf: InlineSection: InlineSection object which point to
        InputMaster.dpmf
        :param components: InlineSection: InlineSection object which point to
        InputMaster.mc
        """
        super(InlineMXSection, self).__init__()
        self.dpmf = dpmf
        self.mc = mc

    def __setattr__(self, key, value):
        OrderedDict.__setattr__(self, key, value)

    def __str__(self):
        """
        Returns string representation of all MX objects contained in this
        InlineMXSection by calling the individual MX object's __str__.
        :return:
        """
        if len(self) == 0:
            return []
        else:
            lines = ["{}_IN_FILE inline".format(self.kind())]
            lines.append(str(len(self)) + "\n")  # total number of MX objects
            for i in range(len(self)):
                lines.append(str(i))  # index of MX object
                # check if number of components is equal to length of DPMF 
                obj = list(self.values())[i]
                dpmf_name = obj.dpmf
                components = obj.components
                dpmf_length = self.dpmf[dpmf_name].get_length()
                if not dpmf_length == len(components):
                    raise ValueError(
                        "Dimension of DPMF must be equal to number of components associated with this MX object.")
                else:
                    lines.append(str(dpmf_length))
                    # dimension of MX
                    lines.append(list(self)[i])  # name of MX
                    lines.append(obj.__str__())
                    # string representation of this MX object

            lines.append("\n")
            return "\n".join(lines)


class InputMaster:
    """
    Master class which contains all GMTK objects present in the input
    master and is responsible for creating their string representation.
    Attributes:
        mean: InlineSection: contains all Mean objects in input master
        covar: InlineSection: contains all Covar objects in input master
        dpmf: InlineSection: contains all DPMF objects in input master
        dense_cpt: InlineSection: contains all DenseCPT objects in input master
        deterministic_cpt: InlineSection: contains all DeterministicCPT objects
        in input master
        mc: InlineMCSection: contains all MC objects in input master
        mx: InlineMXSection: contains all MX objects in input master
        name_collection: InlineSection: contains all NameCollection objects in
        input master
    """

    def __init__(self):
        """
        Initialize InputMaster instance with empty attributes (InlineSection
        and its subclasses).
        """
        self.mean = InlineSection()
        self.covar = InlineSection()
        self.dpmf = InlineSection()
        self.dense_cpt = InlineSection()
        self.deterministic_cpt = InlineSection()
        self.mc = InlineMCSection(mean=self.mean, covar=self.covar)
        self.mx = InlineMXSection(dpmf=self.dpmf, mc=self.mc)
        self.name_collection = InlineSection()

    def __str__(self):
        """
        Return string representation of all the attributes (GMTK types) by
        calling the attributes' (InlineSection and its subclasses) __str__().
        :return:
        """
        attrs = [self.deterministic_cpt, self.name_collection, self.mean,
                 self.covar, self.dense_cpt, self.dpmf, self.mc, self.mx]

        s = []
        for obj in attrs:
            s.append("".join(obj.__str__()))

        return "".join(s)

    def save(self, filename, traindir='segway_output/traindir'):
        """
        Opens filename for writing and writes out the contents of its attributes.
        :param: filename: str: path to input master file 
        :param: traindir: str: path to traindir 
        (default assumes path to traindir is 'segway_output/traindir')
        :return: None 
        """
        with open(filename, 'w') as filename:
            print('# include "' + traindir + '/auxiliary/segway.inc"', file=filename)
            print(self, file=filename)
            

class DenseCPT(Array):
    """
    A single DenseCPT object.
    """
    kind = "DENSE_CPT"

    # todo check if sums to 1.0

    def __str__(self):
        """
        Return string representation of this DenseCPT object.
        :return:
        """
        line = []
        
        num_parents = len(self.shape) - 1
        line.append(str(num_parents))  # number of parents
        cardinality_line = map(str, self.shape)
        line.append(" ".join(cardinality_line))  # cardinalities
        line.append(array2text(self))
        line.append("\n")

        return "\n".join(line)
    
    @classmethod
    def uniform_from_shape(cls, *shape, **kwargs):
        """
        :param: shape: int: shape of DenseCPT
        :param: kwargs: float: optional value for diagonal entry of DenseCPT (default is 0.0) 
        :return: DenseCPT with uniform probabilities and given shape
        """
        if 'self' not in kwargs.keys():
            kwargs['self'] = 0.0  # set default value for diagonal entry to 0
        diag_value = kwargs['self']

        if len(shape) == 1:
            a = np.squeeze(DenseCPT(np.empty(shape)), axis=0)
            a.fill(1.0 / shape[-1])  # uniform by number of columns
            return a
        else:
            a = np.empty(shape)
            div = shape[-1] - 1
   
            # if num_subsegs = 1
            if div == 0:
                a.fill(1.0)
            else:
                value = (1.0 - diag_value) / div
                a.fill(value)
                # len(shape) = 2 => seg_seg => square matrix
                if len(shape) == 2:
                    # replace diagonal entries 
                    diag_index = range(shape[0])
                    a[diag_index, diag_index] = diag_value 

                # len(shape) = 3 => seg_subseg_subseg
                # => num_segs x square matrix
                if len(shape) == 3:
                    # "diag_indices" to be set to 0:
                    # range(seg), diag_index, diag_index
                    diag_index = []
                    for s in range(shape[-1]):
                        diag_index.append([s] * len(shape[1:]))
                    final_indices = []
                    # prepare "diagonal" indices
                    for i in range(shape[0]):
                        for item in diag_index: 
                            index = [i]
                            index.extend(item)
                            final_indices.append(tuple(index))
                    for index in final_indices:  # replace diagonal entries 
                        a[index] = diag_value
                        
            return np.squeeze(DenseCPT(a), axis=0)
        

class NameCollection(list):
    """
    A single NameCollection object.
    """
    kind = "NAME_COLLECTION"

    def __init__(self, *args):
        """
        Initialize a single NameCollection object.
        :param args: str: names in this NameCollection
        """
        if isinstance(args[0], list):  # names in NameCollection have been given in a single list 
            super(NameCollection, self).__init__([])
            self.extend(args[0])
        else:
            super(NameCollection, self).__init__(list(args))

    def __str__(self):
        """
        Returns string format of NameCollection object to be printed into the
        input.master file (new lines to be added)
        """
        if len(self) == 0:
            return ""
        else:
            line = []
            line.append(str(len(self)))
        line.extend(self)
        line.append("\n")
        list.__str__(self)
        return "\n".join(line)


class Mean(Array):
    """
    TODO
    A single Mean object.
    """
    kind = "MEAN"

    def __array_finalize__(self, obj):
        if obj is None: return

    def __str__(self):
        """
        Returns the string format of the Mean object to be printed into the
        input.master file (new lines to be added).
        :return:
        """
        line = []
        line.append(str(self.get_dimension()))  # dimension
        line.append(array2text(self))
        line.append("\n")
        return "\n".join(line)

    def get_dimension(self):
        """
        Return dimension of this Mean object.
        :return: int: dimension of this Mean object
        """
        return len(self)

    
class Covar(Array):
    """
    A single Covar object.
    """
    kind = "COVAR"

    def __str__(self):
        """
        Return string representation of single Covar object.
        :return:
        """
        line = [str(self.get_dimension())] # dimension
        line.append(array2text(self))  # covar values
        line.append("\n")
        return "\n".join(line)

    def get_dimension(self):
        """
        Return dimension of this Covar object.
        :return: int: dimension of this Covar object
        """
        return len(self)
    
    
class DPMF(Array):
    """
    A single DPMF object.
    """
    kind = "DPMF"

    # todo check if sums to 1.0

    def __str__(self):
        """
        Return string representation of this DPMF.
        :return:
        """
        line = [str(self.get_length())]  # dpmf length
        line.append(array2text(self))  # dpmf values
        line.append("\n")
        return "\n".join(line)

    def get_length(self):
        return len(self)
    
    @classmethod
    def uniform_from_shape(cls, *shape):
        """
        :param: shape: int: shape of DPMF 
        :return: DPMF with uniform probabilities and given shape
        """  
        a = np.empty(shape)
        if len(shape) != 1:
            raise ValueError("DPMF must be one-dimensional.")
        else:
            value = 1.0 / shape[0]
            a.fill(value)
            
        return DPMF(a).squeeze(axis=0)

    
class MC:
    """
    A single MC object.
    Attributes:
        component_type: int: type of MC
    """
    kind = "MC"

    def __init__(self, component_type):
        """
        Initialize a single MC object.
        :param component_type: int: type of MC
        """
        self.component_type = component_type


class DiagGaussianMC(MC, object):
    """
    Attributes:
        component_type = 0
        mean: str: name of Mean object associated to this MC
        covar: str: name of Covar obejct associated to this MC
    """
    def __init__(self, mean, covar):
        """
        Initialize a single DiagGaussianMC object.
        :param mean: name of Mean object associated to this MC
        :param covar: name of Covar obejct associated to this MC
        """
        # more component types?
        super(DiagGaussianMC, self).__init__(COMPONENT_TYPE_DIAG_GAUSSIAN)
        self.mean = mean
        self.covar = covar

    def __str__(self):
        """
        Return string representation of this MC object.
        :return:
        """
        return " ".join([self.mean, self.covar])


class MX:
    """
    A single MX object.
    Attributes:
        dpmf: str: name of DPMF object associated with MX
        components: list[str]: names of components associated with this MX
    """
    kind = "MX"

    def __init__(self, dpmf, components):
        """
        Initialize a single MX object.
        :param dpmf: str: name of DPMF object associated with this MX
        :param components: str or list[str]: names of components associated with
        this MX
        """
        self.dpmf = dpmf
        if isinstance(components, str):
            self.components = [components]
        elif isinstance(components, list):
            for name in components:
                if not isinstance(name, str):
                    raise ValueError("All component names must be strings.")
            self.components = components
        else:  # not allowed types
            raise ValueError("Incorrect format of component names.")

    def __str__(self):
        """
        Return string representation of this MX.
        :return:
        """
        line = [str(len(self.components))]  # number of components
        line.append(self.dpmf)  # dpmf name
        line.append(" ".join(self.components))  # component names
        return "\n".join(line)


class DeterministicCPT:
    """
    A single DeterministicCPT object.
    Attributes:
       parent_cardinality: tuple[int]: cardinality of parents
       cardinality: int: cardinality of self
       dt: str: name existing Decision Tree (DT) associated with this
       DeterministicCPT
    """
    kind = "DETERMINISTIC_CPT"

    def __init__(self, cardinality_parents, cardinality, dt):
        """
        Initialize a single DeterministicCPT object.
        :param parent_cardinality: tuple[int]: cardinality of parents
        (if empty, then number of parents = 0
        :param cardinality: int: cardinality of self
        :param dt: name existing Decision Tree (DT) associated with this
       DeterministicCPT
        """
        if not isinstance(cardinality_parents, tuple):
            self.cardinality_parents = (cardinality_parents, )
        else: 
            self.cardinality_parents = cardinality_parents
        self.cardinality = cardinality
        self.dt = dt

    def __str__(self):
        """
        Return string representation of this DeterministicCPT.
        :return:
        """
        line = []
        num_parents = len(self.cardinality_parents)
        line.append(str(num_parents))  # number of parents
        cardinalities = []
        cardinalities.extend(self.cardinality_parents)
        cardinalities.append(self.cardinality)
        line.append(" ".join(map(str, cardinalities)))  # cardinalities of parent and self
        line.append(self.dt)
        line.append("\n")
        return "\n".join(line)
