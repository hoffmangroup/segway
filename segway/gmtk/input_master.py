from __future__ import annotations

from typing import List, Optional, Union

from numpy import array, asarray, empty, ndarray, squeeze

INPUT_MASTER_PREAMBLE = \
    """#define COMPONENT_TYPE_DIAG_GAUSSIAN 0"""


COMPONENT_TYPE_DIAG_GAUSSIAN = 0
OBJ_KIND_MEAN = "MEAN"
OBJ_KIND_COVAR = "COVAR"
OBJ_KIND_NAMECOLLECTION = "NAME_COLLECTION"
OBJ_KIND_DPMF = "DPMF"
OBJ_KIND_DENSECPT = "DENSE_CPT"
OBJ_KIND_DETERMINISTICCPT = "DETERMINISTIC_CPT"
OBJ_KIND_MC = "MC"
OBJ_KIND_MX = "MX"


# "kind" refers to a kind of GMTK object. GMTK often calls these classes
# or objects.
# "class" refers to a Python classs.
# "component type" refers to types of GMTK Mixture Components, such as
# Diagonal Gaussian.


def array2text(a: Array) -> str:
    """
    Convert multi-dimensional array to text.
    """
    if a.ndim == 1:
        return " ".join(map(str, a))
    delimiter = "\n" * (a.ndim - 1)
    return delimiter.join(array2text(row) for row in a)


# Types of array data
NumericArrayLike = Union[float, int, ndarray]


class Array(ndarray):
    def __new__(cls, *args: NumericArrayLike, keep_shape: bool = False) \
            -> Array:
        """
        Create a new Array containing the provided value(s).
        If passed a single value and keep_shape is False, insert a new first
        dimension of size 1. This is needed for creating vector arrays
        (dimension 1) from a single scalar input value.
        """
        # Ensure all arguments belong to the correct type
        if not all(isinstance(arg, NumericArrayLike) for arg in args):
            # If union iterable, fix. Otherwise, hardwrite
            raise TypeError("Argument has incompatible type. "
                            f"Expected {NumericArrayLike}")

        input_array = array(args)

        # Create a new array object with the same data as input_array
        # but with the type of cls (i.e. Array)
        res = asarray(input_array).view(cls)

        # If keep_shape is true, remove the new dimension
        if len(args) == 1 and keep_shape:
            res = squeeze(res, axis=0)

        return res


class DenseCPT(Array):
    """
    A single DenseCPT object.
    """
    kind = OBJ_KIND_DENSECPT

    # todo check if sums to 1.0

    def __str__(self) -> str:
        """
        Return string representation of this DenseCPT object.
        """
        return f"{array2text(self)}\n"

    def get_header_info(self) -> str:
        """
        Return number of parents, cardinality line, for header in
        input.master section.
        """
        line = [str(len(self.shape) - 1)]  # number of parents
        cardinality_line = map(str, self.shape)
        line.append(" ".join(cardinality_line))  # cardinalities
        return " ".join(line)

    @classmethod
    def uniform_from_shape(cls, *shape: int,
                           self_transition: float = 0.0) -> DenseCPT:
        """
        Create a Dense CPT with the specified self_transition probability
        (diagonal entries) and a uniform probability for all other
        transitions (non-diagonal entries).

        :param: shape: int: shape of DenseCPT
        :param: self_transition: float: optional value for diagonal entry of
        DenseCPT (default is 0.0)
        :return: DenseCPT with given shape and specified probabilities
        """
        values = empty(shape)
        if len(shape) == 1:
            values.fill(1.0 / shape[-1])
        # Check if num_subsegs == 1 to prevent divide-by-zero error
        elif (shape[-1] - 1) == 0:
            values.fill(1.0)
        else:
            value = (1.0 - self_transition) / (shape[-1] - 1)
            values.fill(value)

            # len(shape) = 2 => seg_seg
            # => square matrix of num_seg x num_seg
            if len(shape) == 2:
                # set diagonal elements to self_transition
                diag_index = range(shape[0])
                values[diag_index, diag_index] = self_transition

            # len(shape) = 3 => seg_subseg_subseg
            # => num_segs x (square matrix of num_subseg x num_subseg)
            if len(shape) == 3:
                # In a 3 dimensional CPT, for each segment's square matrix,
                # set the diagonal entries to self_transition.
                # Begin by creating the list of indices for diagonal entries
                # in a segment's square matrix
                diag_index = []
                for subseg_index in range(shape[-1]):
                    diag_index.append([subseg_index] * len(shape[1:]))

                # For each segment, create the full indices for the diagonal
                # entries within its square matrix
                final_indices = []
                for seg_index in range(shape[0]):
                    for subseg_indices in diag_index:
                        index = [seg_index]
                        index.extend(subseg_indices)
                        final_indices.append(tuple(index))

                # Set all diagonal entries of each segment's square matrix
                # to self_transition
                for index in final_indices:
                    values[index] = self_transition

        return DenseCPT(values, keep_shape=True)


class NameCollection(list):
    """
    A single NameCollection object.
    """
    kind = OBJ_KIND_NAMECOLLECTION

    def __init__(self, names: List[str]):
        """
        Initialize a single NameCollection object.
        :param names: List[str]: names in this NameCollection
        """
        # Ensure all arguments are strings or lists of strings
        if not all(isinstance(name, str) for name in names):
            raise TypeError("All arguments must be str instances.")

        super().__init__(names)

    def __str__(self) -> str:
        """
        Returns string format of NameCollection object to be printed into the
        input.master file (new lines to be added)
        """
        return "\n".join(list(self))

    def get_header_info(self) -> str:
        """
        Return size of NameCollection, for header in input.master section.
        """
        return str(len(self))


class OneLineKind(Array):
    """
    An abstract Python class acting as the parent for Python classes
    representing Array-like GMTK kinds that have one-line string
    representations in input.master.
    It inherits the Array data structure for data storage and defines
    the __str__ method for writing this data as one line.
    """

    def get_header_info(self) -> str:
        """
        Return string representation of own information, for header in
        input.master section.
        """
        line = [str(len(self))]  # dimension
        line.append(array2text(self))  # array values
        return " ".join(line)


class Mean(OneLineKind):
    """
    A single Mean object.
    __init__ and __str__ methods defined in superclass.
    """
    kind = OBJ_KIND_MEAN


class Covar(OneLineKind):
    """
    A single Covar object.
    __init__ and __str__ methods defined in superclass.
    """
    kind = OBJ_KIND_COVAR


# These types must be defined before they are referenced, so these constants
# are defined here rather than the top of the file.

# object classes that can support conversion
CONVERTIBLE_CLASSES = {
    OBJ_KIND_NAMECOLLECTION: NameCollection,
    OBJ_KIND_MEAN: Mean,
    OBJ_KIND_COVAR: Covar
}

# Types which can be auto-converted by convert
ConvertableGMTKObjectType = Union[tuple(cls for cls in
                                        CONVERTIBLE_CLASSES.values())]


def convert(
        cls: ConvertableGMTKObjectType,
        value: Union[NumericArrayLike, ConvertableGMTKObjectType, List[str],
                     str]
) -> ConvertableGMTKObjectType:
    """
    Convert a provided attribute of input_master which a non-GMTK object into
    a GMTK object of the provided class.
    :param cls: GMTK class
    :param value: Value to convert to GMTK object
    """
    # If provided value matches the desired class, no conversion needed
    if isinstance(value, cls):
        return value  # type: ignore[return-value]

    # Otherwise, create new object of with provided data
    return cls(value)


class DPMF(OneLineKind):
    """
    A single DPMF object.
    """
    kind = OBJ_KIND_DPMF

    # todo check if sums to 1.0

    @classmethod
    def uniform_from_shape(cls, shape: int) -> DPMF:
        """
        :param: shape: int: shape of DPMF, which must be 1-dimensional.
        :return: DPMF with uniform probabilities and given shape
        """
        dpmf_values = empty(shape)
        value = 1.0 / shape
        dpmf_values.fill(value)

        return DPMF(dpmf_values, keep_shape=True)


class MC:
    """
    A single mixture component (MC) object.
    Attributes:
        component_type: str: type of MC
    """
    kind = OBJ_KIND_MC

    def __init__(self, component_type: str):
        """
        Initialize a single MC object.
        :param component_type: str: type of MC, such as
            COMPONENT_TYPE_DIAG_GAUSSIAN
        """
        self.component_type = component_type

    def get_header_info(self) -> str:
        # No additional header information needed in input.master.
        return ""


class DiagGaussianMC(MC, object):
    """
    Attributes:
        component_type = 0
        mean: str: name of Mean object associated to this MC
        covar: str: name of Covar obejct associated to this MC
    """
    def __init__(self, mean: str, covar: str):
        """
        Initialize a single DiagGaussianMC object.
        :param mean: name of Mean object associated to this MC
        :param covar: name of Covar obejct associated to this MC
        """
        # more component types?
        super().__init__("COMPONENT_TYPE_DIAG_GAUSSIAN")
        self.mean = mean
        self.covar = covar

    def __str__(self) -> str:
        """
        Return string representation of this MC object.
        """
        return " ".join([self.mean, self.covar])


class MX:
    """
    A single mixture (MX) object.
    Attributes:
        dpmf: str: name of DPMF object associated with MX
        components: list[str]: names of components associated with this MX
    """
    kind = OBJ_KIND_MX

    def __init__(self, dpmf: str, components: Union[str, List[str]]):
        """
        Initialize a single MX object.
        :param dpmf: str: name of DPMF object associated with this MX
        :param components: str or list[str]: names of components associated
        with this MX
        """
        self.dpmf = dpmf
        if isinstance(components, str):
            self.components = [components]
        elif isinstance(components, list):
            if not all(isinstance(name, str) for name in components):
                raise ValueError("Each component must be str")
            self.components = components
        else:  # not allowed types
            raise ValueError("`components` must be str or List[str].")

    def __str__(self) -> str:
        """
        Return string representation of this MX.
        """
        line = [str(len(self.components))]  # number of components
        line.append(self.dpmf)  # dpmf name
        line.append(" ".join(self.components))  # component names
        return " ".join(line)

    def get_header_info(self) -> str:
        # No additional header information needed in input.master.
        return ""


class DeterministicCPT:
    """
    A single DeterministicCPT object.
    Attributes:
       parent_cardinality: tuple[int]: cardinality of parents
       cardinality: int: cardinality of self
       dt: str: name existing Decision Tree (DT) associated with this
       DeterministicCPT
    """
    kind = OBJ_KIND_DETERMINISTICCPT

    def __init__(self, cardinality_parents: Union[tuple[int], int],
                 cardinality: int, dt: str):
        """
        Initialize a single DeterministicCPT object.
        :param cardinality_parents: tuple[int]: cardinality of parents
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

    def __str__(self) -> str:
        """
        Return string representation of this DeterministicCPT.
        """
        line = [str(len(self.cardinality_parents))]  # lines
        cardinalities = list(self.cardinality_parents)
        cardinalities.append(self.cardinality)

        # cardinalities of parent and self
        line.append(" ".join(map(str, cardinalities)))
        line.append(f"{self.dt}\n")
        return "\n".join(line)

    def get_header_info(self) -> str:
        """
        No additional header information needed in input.master.
        """
        return ""


class Section(dict):
    """
    Contains GMTK objects of a single type and supports writing them to file.
    Key: name of GMTK object
    Value: GMTK object
    Attributes:
            kind: str: specifies the kind of GMTK object
            (default assumes that `self` has no kind)
    """
    def __init__(self, kind: Optional[str] = None):
        """
        Initialize an empty Section object.
        """
        super().__init__()
        self.kind = kind

    def __setitem__(
            self,
            key: str,
            value: Union[float, int, List[str], str,
                         Mean, Covar, NameCollection,
                         DPMF, DiagGaussianMC, MX, DenseCPT]):
        cls = CONVERTIBLE_CLASSES.get(self.kind)
        if cls is not None:
            value = convert(cls, value)

        # self.kind is undefined for objects that don't support type conversion
        if not self.kind:
            # Set self.kind as the kind of first GMTK type value passed
            # consistency of kind for all values are checked in InlineSection
            # as the whole dictionary could be checked at once
            self.kind = value.kind

        dict.__setitem__(self, key, value)

    def get_header_lines(self) -> List[str]:
        """
        Generate header lines for this Section object.
        """
        # object title and total number of GMTK objects
        return [f"{self.kind}_IN_FILE inline", f"{len(self)}\n"]


class InlineSection(Section):

    def __str__(self) -> str:
        """
        Return inline string representation of this Section object by calling
        the individual GMTK object's `__str__()`.
        """
        # Assert all the values in the dict are of the expected type
        assert all(obj.kind == self.kind
                   for obj in self.values()), "Objects must be of same type."

        # if this section stores no GMTK objects
        if len(self) == 0:
            return ""

        # The section generates the index and name of GMTK object
        lines = self.get_header_lines()
        for index, (key, value) in enumerate(self.items()):
            # Use section information to generate index and name of GMTK object
            obj_header = [str(index), key]

            # Use GMTK object to generate additional special header information
            # Append this to the index and name above
            obj_header.append(value.get_header_info())

            # Use rstrip to remove the trailing space for GMTK types with
            # no additional header information
            lines.append(" ".join(obj_header).rstrip())

            # One line kind objects have all data included in the header
            # If not one line kind, write the object's remaining data lines
            if not isinstance(value, OneLineKind):
                lines.append(str(value))

        return "\n".join(lines + [""])


class InlineMCSection(InlineSection):
    """
    Special InlineSection subclass which contains MC objects.
    Attributes:
        mean: the InlineSection object stored at InputMaster.mean
        covar: the InlineSection object stored at InputMaster.covar
    """
    def __init__(self, mean: InlineSection, covar: InlineSection):
        """
        :param mean: InlineSection: the InlineSection object stored at
        InputMaster.mean
        :param covar: InlineSection: the InlineSection object stored at
        InputMaster.covar
        """
        super().__init__(OBJ_KIND_MC)
        self.mean = mean
        self.covar = covar

    def __str__(self) -> str:
        """
        Returns string representation of all MC objects contained in this
        InlineMCSection by calling the individual MC object's `__str__()`.
        """
        if len(self) == 0:
            return ""

        lines = self.get_header_lines()
        for index, (name, obj) in enumerate(list(self.items())):
            # check if dimension of Mean and Covar of this MC are the same
            mean_ndim = len(self.mean[obj.mean])
            covar_ndim = len(self.covar[obj.covar])
            if mean_ndim != covar_ndim:
                raise ValueError("Inconsistent dimensions of mean and covar.")

            obj_line = [str(index)]  # index of MC object
            obj_line.append(str(mean_ndim))  # MC dimension
            obj_line.append(str(obj.component_type))
            obj_line.append(name)  # name of MC

            # string representation of MC obj
            obj_line.append(str(obj))
            lines.append(" ".join(obj_line))

        return "\n".join(lines + [""])


class InlineMXSection(InlineSection):
    """
    Special InlineSection subclass which contains MX objects.
    Attributes:
        dpmf: the InlineSection object stored at InputMaster.dpmf
    """

    def __init__(self, dpmf: InlineSection):
        """
        :param dpmf: InlineSection: the InlineSection object stored at
        InputMaster.dpmf
        """
        super().__init__(OBJ_KIND_MX)
        self.dpmf = dpmf

    def __str__(self) -> str:
        """
        Returns string representation of all MX objects contained in this
        InlineMXSection by calling the individual MX object's `__str__()`.
        """
        if len(self) == 0:
            return ""

        lines = self.get_header_lines()
        for index, (name, obj) in enumerate(list(self.items())):
            # Assert number of components is equal to length of DPMF
            dpmf_ndim = len(self.dpmf[obj.dpmf])
            if not dpmf_ndim == len(obj.components):
                raise ValueError(
                    "Dimension of DPMF must be equal to the "
                    "number of components associated with this MX object.")

            obj_line = [str(index)]  # index of MX object
            obj_line.append(str(dpmf_ndim))  # dimension of MX
            obj_line.append(name)  # name of MX
            obj_line.append(str(obj))

            # string representation of this MX object
            lines.append(" ".join(obj_line))

        return "\n".join(lines + [""])


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
        self.deterministic_cpt = InlineSection(OBJ_KIND_DETERMINISTICCPT)
        self.name_collection = InlineSection(OBJ_KIND_NAMECOLLECTION)
        self.mean = InlineSection(OBJ_KIND_MEAN)
        self.covar = InlineSection(OBJ_KIND_COVAR)
        self.dense_cpt = InlineSection(OBJ_KIND_DENSECPT)
        self.dpmf = InlineSection(OBJ_KIND_DPMF)
        self.mc = InlineMCSection(mean=self.mean, covar=self.covar)
        self.mx = InlineMXSection(dpmf=self.dpmf)

    def __str__(self) -> str:
        """
        Return string representation of all the attributes (GMTK types) by
        calling the attributes' (InlineSection and its subclasses) `__str__()`.
        """
        sections = [self.deterministic_cpt, self.name_collection, self.mean,
                    self.covar, self.dense_cpt, self.dpmf, self.mc, self.mx]

        return "\n".join(str(section) for section in sections)

    def save(self, filename: str) -> None:
        """
        Write the specified InputMaster object as a string representation to
        the provided filename
        Open filename for writing and write the string representation
        :param: filename: str: path to input master file
        """
        with open(filename, "w") as outfile:
            print(INPUT_MASTER_PREAMBLE, file=outfile)
            print(self, file=outfile)
