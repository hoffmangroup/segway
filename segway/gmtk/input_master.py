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
OBJ_KIND_DT = "DT"
OBJ_KIND_VECPT = "VE_CPT"
OBJ_KIND_ARBITRARYSTRING = "ARBITRARY_STRING"
OBJ_KIND_RM = "REAL_MAT"
OBJ_KIND_DIRICHLETTAB = "DIRICHLET_TAB"


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
            raise TypeError("Argument has incompatible type."
                            f"Expected {NumericArrayLike}")

        input_array = array(args)

        # Create a new array object with the same data as input_array
        # but with the type of cls (i.e. Array)
        res = asarray(input_array).view(cls)

        # If keep_shape is true, remove the new dimension
        if len(args) == 1 and keep_shape:
            res = squeeze(res, axis=0)

        return res


class MultiDimArray(Array):
    """
    Abstract class providing features for multidimensional arrays which are
    represented on multiple lines.
    """
    
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


class DenseCPT(MultiDimArray):
    """
    A single DenseCPT object.
    """
    kind = OBJ_KIND_DENSECPT

    # todo check if sums to 1.0

    def set_dirichlet_table(self, dirichlet_name) -> str:
        """
        Set the name of a Dirichlet table which this CPT will reference.
        """
        self.dirichlet_name = dirichlet_name

    def get_header_info(self) -> str:
        """
        Return multidimensional array header with dimensions.
        If the dirichlet_name attribute was set, include an additional line
        in the header referencing the DirichletTable with that name.
        """
        line = MultiDimArray.get_header_info(self)
        if hasattr(self, "dirichlet_name"):
            dirichlet_line = f"DirichletTable dirichlet_{self.dirichlet_name}"
            line = "\n".join((line, dirichlet_line))
        return line

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


class DirichletTable(MultiDimArray):
    """
    A single DirichletTable object.
    """
    kind = OBJ_KIND_DIRICHLETTAB

    def get_header_info(self) -> str:
        """
        Return number of parents, cardinality line, for header in
        input.master section.
        """
        line = [str(len(self.shape))]  # number of parents
        cardinality_line = map(str, self.shape)
        line.append(" ".join(cardinality_line))  # cardinalities
        return " ".join(line)


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


class RealMat(OneLineKind):
    """
    An entry in a Real matrix object.
    __init__ and __str__ methods defined in superclass.
    """
    kind = OBJ_KIND_RM

    def get_header_info(self) -> str:
        """
        Return string representation of own information, for header in
        input.master section.
        """
        # TODO: Should the second 1 be hardcoded? Or is this 2D data?
        line = [str(len(self)), "1"]  # dimension, additional value
        line.append(array2text(self))  # array values
        return " ".join(line)


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

    # XXX: check if sums to 1.0

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
        :param component_type: str: type of MC
        """
        self.component_type = component_type

    def get_header_info(self) -> str:
        # No additional header information needed in input.master.
        return ""


class DiagGaussianMC(MC, object):
    """
    Attributes:
        component_type = "COMPONENT_TYPE_DIAG_GAUSSIAN" = 0
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
    

class MissingFeatureDiagGaussianMC(MC, object):
    """
    Attributes:
        component_type = "COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN"
        mean: str: name of Mean object associated to this MC
        covar: str: name of Covar obejct associated to this MC
    """
    def __init__(self, mean: str, covar: str):
        """
        Initialize a single MissingFeatureDiagGaussianMC object.
        :param mean: name of Mean object associated to this MC
        :param covar: name of Covar obejct associated to this MC
        """
        # more component types?
        super().__init__("COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN")
        self.mean = mean
        self.covar = covar

    def __str__(self) -> str:
        """
        Return string representation of this MC object.
        """
        return " ".join([self.mean, self.covar, "matrix_weightscale_1x1"])


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

    def __init__(self, cardinality_parents: Union[tuple[str], str],
                 cardinality: str, dt: str):
        """
        Initialize a single DeterministicCPT object.
        :param cardinality_parents: tuple[str]: cardinality of parents
        (if empty, then number of parents = 0)
        :param cardinality: str: cardinality of self
        :param dt: str: name existing Decision Tree (DT) associated with
        this DeterministicCPT
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
        line = [str(len(self.cardinality_parents))]  # number of parents
        cardinalities = list(self.cardinality_parents)
        cardinalities.append(self.cardinality)

        # cardinalities of parent and self
        line.append(" ".join(map(str, cardinalities)))
        line.append(f"{self.dt}\n")
        return "\n".join(line)

    def get_header_info(self) -> str:
        # No additional header information needed in input.master.
        return ""


class ArbitraryString:
    """
    A class storing an arbitrary string to write to input.master.
    Attributes:
        contents: str: Arbitrary string to write to input.master
    """
    kind = OBJ_KIND_ARBITRARYSTRING

    def __init__(self, contents: str):
        """
        Initialize an arbitrary string class.
        :param contents: arbitrary string to write to input.master
        """
        self.contents = contents

    def __str__(self) -> str:
        """
        Return the stored string.
        """
        return self.contents

    def get_header_info(self) -> str:
        # No additional header information
        return ""


class DecisionTree(ArbitraryString):
    """
    A Decision Tree object.
    """
    kind = OBJ_KIND_DT

    def __init__(self, tree: str):
        """
        Initialize a DecisionTree object.
        :param tree: String representation of the tree
        """
        super().__init__(tree)


class Section(dict):
    """
    Contains GMTK objects of a single type and supports writing them to file.
    Key: name of GMTK object
    Value: GMTK object
    Attributes:
            kind: str: specifies the kind of GMTK object (default assumes that
                `self` has no kind)
            line_before: str: string to print before the section, often a
                preprocessor rule
            line_after: str: string to print after the section, often a
                preprocessor rule
    """
    def __init__(self, kind: Optional[str] = None):
        """
        Initialize an empty Section object.
        """
        super().__init__()
        self.kind = kind
        self.line_before = None
        self.line_after = None

    def __setitem__(
            self,
            key: str,
            value: Union[float, int, List[str], str,
                         Mean, Covar, NameCollection,
                         DPMF, DiagGaussianMC, MX, DenseCPT]):
        cls = CONVERTIBLE_CLASSES.get(self.kind)
        if cls is not None:
            value = convert(cls, value)

        # self.kind is undefined for objects that dont support type conversion
        if not self.kind:
            # sets self.kind as the kind of first GMTK type value passed
            # consistency of kind for all values are checked in InlineSection
            # as the whole dictionary could be checked at once
            self.kind = value.kind

        dict.__setitem__(self, key, value)

    def get_header_lines(self) -> List[str]:
        """
        Generate header lines for this Section object.
        """
        # object title and total number of GMTK/MC/MX objects
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
        
        # if line_before is set, use it to begin the section's lines
        lines = []
        if self.line_before is not None:
            lines += [self.line_before]
        
        # if stored items are arbitrary strings, return them out without
        # any additional formatting. Otherwise, apply formatting
        if self.kind == OBJ_KIND_ARBITRARYSTRING:
            lines += self.get_unformatted_lines()
        else:
            lines += self.get_formatted_lines()

        # if line_after is set, use it to end the section's lines
        if self.line_after is not None:
            lines += [self.line_after]

        return "\n".join(lines + [""])
    
    def get_formatted_lines(self) -> List[str]:
        """
        Format the GMTK objects with a section header and object headers 
        """
        lines = self.get_header_lines()
        for index, (key, value) in enumerate(self.items()):
            obj_header = [str(index), key]  # Index and name of GMTK object

            # Special header information for some GMTK types
            obj_header.append(value.get_header_info())

            # Use rstrip to remove the trailing space for GMTK types with
            # no additional header information
            lines.append(" ".join(obj_header).rstrip())

            # If not one line kind, write the object's remaining lines
            if not isinstance(value, OneLineKind):
                lines.append(str(value))
        
        return lines
    
    def get_unformatted_lines(self) -> List[str]:
        """
        Extract the string representation of all GMTK objects, with no
        headers or additional formatting. Intended for representing
        Arbitrary String objects.
        """
        return [str(value) for value in self.values()]


class InlineMCSection(InlineSection):
    """
    Special InlineSection subclass which contains MC objects.
    Attributes:
        mean: InlineSection object which point to InputMaster.mean
        covar: InlineSection object which point to InputMaster.covar
    """
    def __init__(self, mean: InlineSection, covar: InlineSection):
        """
        :param mean: InlineSection: InlineSection object which point to
        InputMaster.mean
        :param covar: InlineSection: InlineSection object which point to
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
        
        # if line_before is set, use it to begin the section's lines
        lines = []
        if self.line_before is not None:
            lines += [self.line_before]

        lines += self.get_header_lines()
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

        # if line_after is set, use it to end the section's lines
        if self.line_after is not None:
            lines += [self.line_after]

        return "\n".join(lines + [""])


class InlineMXSection(InlineSection):
    """
    Special InlineSection subclass which contains MX objects.
    Attributes:
        dpmf: InlineSection object which point to InputMaster.dpmf
        components: InlineSection object which point to InputMaster.mc
    """

    def __init__(self, dpmf: InlineSection, mc: InlineSection):
        """
        :param dpmf: InlineSection: InlineSection object which point to
        InputMaster.dpmf
        :param components: InlineSection: InlineSection object which point to
        InputMaster.mc
        """
        super().__init__(OBJ_KIND_MX)
        self.dpmf = dpmf
        self.mc = mc

    def __str__(self) -> str:
        """
        Returns string representation of all MX objects contained in this
        InlineMXSection by calling the individual MX object's `__str__()`.
        """
        if len(self) == 0:
            return ""
        
        # if line_before is set, use it to begin the section's lines
        lines = []
        if self.line_before is not None:
            lines += [self.line_before]

        lines += self.get_header_lines()
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

        # if line_after is set, use it to end the section's lines
        if self.line_after is not None:
            lines += [self.line_after]

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

    def __init__(self, preamble=INPUT_MASTER_PREAMBLE):
        """
        Initialize InputMaster instance with empty attributes (InlineSection
        and its subclasses).
        """
        self.preamble = preamble
        self.dt = InlineSection(OBJ_KIND_DT)
        self.name_collection = InlineSection(OBJ_KIND_NAMECOLLECTION)
        self.dirichlet = InlineSection(OBJ_KIND_DIRICHLETTAB)
        self.deterministic_cpt = InlineSection(OBJ_KIND_DETERMINISTICCPT)
        self.virtual_evidence = InlineSection(OBJ_KIND_ARBITRARYSTRING)
        self.dense_cpt = InlineSection(OBJ_KIND_DENSECPT)
        self.mean = InlineSection(OBJ_KIND_MEAN)
        self.covar = InlineSection(OBJ_KIND_COVAR)
        self.dpmf = InlineSection(OBJ_KIND_DPMF)
        self.mc = InlineMCSection(mean=self.mean, covar=self.covar)
        self.mx = InlineMXSection(dpmf=self.dpmf, mc=self.mc)
        self.real_mat = InlineSection(OBJ_KIND_RM)

    def __str__(self) -> str:
        """
        Return string representation of all the attributes (GMTK types) by
        calling the attributes' (InlineSection and its subclasses) `__str__()`.
        """
        sections = [self.preamble, self.dt, self.name_collection,
                    self.dirichlet, self.deterministic_cpt,
                    self.virtual_evidence, self.dense_cpt, self.mean,
                    self.covar, self.dpmf, self.mc, self.mx, self.real_mat]

        return "\n".join([str(section) for section in sections])

    def save(self, filename: str) -> None:
        """
        Opens filename for writing and writes out
        the contents of its attributes.
        :param: filename: str: path to input master file
        (default assumes path to `traindir` is "traindir")
        """
        with open(filename, "w") as file:
            print(self, file=file)
