====================
GMTK API Reference
====================

Description
===========

The GMTK API provides a Python framework to design dynamic Bayesian models for custom 
tasks. Segway can perform training and inference using these user-defined models.

The GMTK API is installed with Segway automatically. 

Workflow
========

The GMTK API describes the model using two files:

  1. A structure file describing the graphical network's nodes and 
  edges. This should be written using the GMTK structure language. The GMTK 
  structure language is described in the 
  `GMTK Documentation <https://github.com/melodi-lab/gmtk/blob/master/documentation.pdf>`_.

  2. A Python file describing the initial parameter settings and Segway 
  commands for training and inference. The classes provided by the GMTK API
  for writing this file are described below. It should contain 3 sections:

    1. Code defining an :py:class:`InputMaster` object and setting its 
    appropriate attributes to describe the model.

    2. A line saving the :py:class:`InputMaster` object to an input file, 
    which describes the parameters in GMTK structure format. 

    3. Code calling Segway to train the defined model and annotate using the 
    trained model. More information on running Segway is available in 
    :ref:`python-interface`.

The `CNVway code <https://github.com/hoffmangroup/cnvway>`_ provides a worked 
example applying the GMTK API to defining, training, and running a new model.

.. todo: other section? flip sentence order? link?

InputMaster Class
=================

Central class storing all parameter information.

.. py:class:: InputMaster

    .. py:attribute:: name_collection
        :type: InlineSection
    
    Stores the names of distributions and their state names. 
    
    Behaves as a dictionary where keys are distribution names, which 
    can be referenced in the structure file, and each value should be set to 
    a list of that distribution's state names or a :py:class:`NameCollection` 
    object initialized with state names. If a Python list is given, it is 
    converted to a :py:class:`NameCollection` object using 
    :py:meth:`NameCollection.__init__`.

    .. py:attribute:: mean
        :type: InlineSection

    Stores the mean parameters for named distributions. 
    
    Behaves as a dictionary where keys are distribution names and each
    value should be set to the mean value or a :py:class:`Mean` object 
    initialized with the mean value. If a Python float or list of floats is 
    given, it is converted to a :py:class:`Mean` object using 
    :py:meth:`Mean.__init__`.

    .. py:attribute:: covar
        :type: InlineSection

    Stores the covariance parameters for named distributions.
    
    Behaves as a dictionary where keys are distribution names and each
    value should be set to the covariance value or a :py:class:`Covar` object 
    initialized with the covariance value. If a Python float or list of floats is 
    given, it is converted to a :py:class:`Covar` object using 
    :py:meth:`Covar.__init__`.

    .. py:attribute:: dpmf
        :type: InlineSection

    Stores dense probability mass function (DPMF) objects, which can later be used
    to define Gaussian Mixture models. 
    
    Behaves as a dictionary where keys are distribution names and each value 
    should be set to a :py:class:`DPMF` object. 

    .. py:attribute:: mc
        :type: InlineMCSection
        :value: InlineMCSection(mean = self.mean, covar = self.covar)

    Stores Gaussians acting as mixture components (MC) for a Gaussian mixture
    model.
    
    Behaves as a dictionary where keys are distribution names and each value 
    should be set to an :py:class:`MC` object.

    .. py:attribute:: mx
        :type: InlineMXSection
    
    Store Gaussian mixture (mx) distributions constructed from above-defined mixture 
    components and dense probability mass functions.
    
    Behaves as a dictionary where keys are distribution names, usually 
    corresponding to hidden state names of an emission variable (from 
    :py:attr:`self.name_collection`) and each value is an :py:class:`MX` object.

    .. py:attribute:: dense_cpt
        :type: InlineSection

    Stores dense conditional probability tables (CPTs) used in the model. 
    
    Behaves as a dictionary where keys are distribution names, which can 
    be referenced in the structure file, and each value is a 
    :py:class:`DenseCPT` object.  

    .. py:attribute:: deterministic_cpt
        :type: InlineSection

    Stores deterministic conditional probability tables (CPTs) used in the model.
    
    Behaves as a dictionary where keys are distribution names, which can 
    be referenced in the structure file, and each value is a 
    :py:class:`DeterministicCPT` object.

    .. py:method:: __init__(self)

        Create an `InputMaster` object where all attributes are empty.

    .. py:method:: save(self, filename)

        Save all parameters to the provided file, for Segway to use in training
        and annotation.

        :param filename: Path to input master file, where results are saved
        :type filename: str
        :returns: None
        :rtype: None
    
Usage example:

.. code-block:: python

    # Create InputMaster object
    input_master = InputMaster()
    # Set parameters
    ...
    # Save to output file
    input_master.save("input.master")


Parameter Classes
=================

Class representing user-defined model parameters.

.. py:class:: NameCollection

    A list of names with a specialized string method for writing to the 
    parameter file.

    .. py:method:: __init__(self, names)

        Create a :py:class:`NameCollection` object containing the provided names.

        :param names: List of names
        :type names: list[str]

Usage example:

.. code-block:: python

    # Create a NameCollection object in the InputMaster 
    # name_collection InlineSection
    input_master.name_collection["labels"] = \
        NameCollection(["label1", "label2"])
    # Alternately, a list will be converted to a NameCollection
    input_master.name_collection["labels"] = ["label1", "label2"]


.. py:class:: Mean

    A Numpy ``ndarray`` representing a distribution's mean, with a specialized 
    string method for writing to the parameter file. Supports monovariate 
    and multivariate distributions.

    .. py:method:: __init__(self, *args)

        Create a :py:class:`Mean` object storing the provided mean value or 
        vector.

        :param args: The mean value which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

Usage example:

.. code-block:: python

    # Create a Mean object in the InputMaster mean InlineSection
    input_master.mean["dist1"] = Mean(0.0)
    # Alternately, a numeric value will be converted to a Mean
    input_master.mean["dist2"] = 0.0


.. py:class:: Covar

    A Numpy ``ndarray`` representing a distribution's covariance, with a 
    specialized string method for writing to the parameter file. Supports 
    monovariate and multivariate distributions.

    .. py:method:: __init__(self, *args)

        Create a :py:class:`Covar` object storing the provided covariance 
        value or vector.

        :param args: The covariance value which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

Usage example:

.. code-block:: python

    # Create a Covar object in the InputMaster covar InlineSection
    input_master.covar["dist1"] = Covar(1.0)
    # Alternately, a numeric value will be converted to a Covar
    input_master.covar["dist2"] = 1.0


.. py:class:: DPMF

    A Numpy ``ndarray`` representing a dense probability mass function (DPMF) 
    with a specialized string method for writing to the parameter file. As it 
    is intended for use in Gaussian mixture models, it supports monovariate 
    distributions only. 

    .. py:method:: __init__(self, *args)

        Create a :py:class:`DPMF` object storing the provided distribution.

        :param args: The probability distribution as an array of probabilties which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like or multiple arguments

    .. py:classmethod:: uniform_from_shape(self, shape)

        A class method for creating a uniform DPMF with the specified shape.

        :param shape: The shape of the DPMF, as its integer length.
        :type shape: int
        :returns: DPMF with given shape and uniform probabilities.
        :rtype: DPMF

Usage example:

.. code-block:: python

    # Create a custom DPMF object in the InputMaster mean InlineSection
    input_master.dpmf["biased"] = DPMF([0.7, 0.3])
    # Create a uniform DPMF with a specified shape
    input_master.dpmf["uniform"] = DPMF.uniform_from_shape(3)


.. py:class:: DiagGaussianMC

    A Gaussian distribution with a diagonal covariance matrix, for use as a 
    mixture component (MC) in a Gaussian mixture model. Currently the only 
    concrete MC subclass. 

    .. py:attribute:: mean
        :type: str

        Name of a :py:class:`Mean` object representing the mean of this 
        Gaussian.
    
    .. py:attribute:: covar
        :type: str
        
        Name of a :py:class:`Covar` object representing the covariance 
        vector along the diagonal of the covariance matrix. 

    .. py:method:: __init__(self, mean, covar)

        Create a :py:class:`DiagGaussianMC` object with the specified mean 
        and covariance.

        :param mean: Name of a Mean object for the distribution mean
        :type mean: str
        :param covar: Name of a Covar object for the diagonal covariance vector of the distribution
        :type covar: str

Usage example:

.. code-block:: python

    # Create a DiagGaussian object in the InputMaster mc 
    # InlineMCSection.
    # Arguments are labels for Mean and Covariance objects.
    input_master.mc["dist1"] = \
        DiagGaussianMC(mean = "dist1", covar = "dist1")
    input_master.mc["dist2"] = \
        DiagGaussianMC(mean = "dist2", covar = "dist2")


.. py:class:: MX

    A Gaussian mixture (MX) model built from Gaussian mixture components.

    .. py:attribute:: dpmf
        :type: str

        Name of a dense probabiliy mass function :py:class:`DPMF` object 
        representing the contribution of each Gaussian mixture component to 
        the mixture model.
    
    .. py:attribute:: components
        :type: str or list[str]

        Names of Gaussian components associated with the mixture model. 

    .. py:method:: __init__(self, dpmf, components)

        Create an :py:class:`MX` object with the mixture distribution and 
        components.

        :param dpmf: Name of a DPMF describing mixture weights.
        :type dpmf: str
        :param components: Name or list of names of mixture components
        :type components: str or list[str]

Usage example:

.. code-block:: python

    # Create a MX objects in the InputMaster mx InlineMXSection.
    # Arguments are labels for DPMF and MX objects.
    input_master.mx["emission1"] = MX("biased", ["dist1", "dist2"])
    input_master.mx["emission2"] = MX("biased", ["dist1", "dist2"])


.. py:class:: DenseCPT

    A Numpy ``ndarray`` representing a dense conditional probability table 
    (CPT) with a specialized string method for writing to the parameter file. 
    Supports up to 3 dimensional tables.
    
    .. py:method:: __init__(self, *args)

        Create a :py:class:`DenseCPT` object storing the provided distribution.

        :param args: The probability distribution as an array of probabilties which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

    .. py:classmethod:: uniform_from_shape(*shape, self=0.0)

        A class method for creating a :py:class:`DenseCPT` object with the provided 
        shape.
        If the table is 2 or 3 dimensional, the diagonal entries of the table 
        are set to the ``self_transition`` parameter (default 0.0) and all other 
        entries are set to be uniform. 

        :param shape: Shape of Dense CPT table
        :type shape: Array_like or multiple arguments
        :param self_transition: Value for diagonal entries in the table. Defaults to 0.0
        :type self: float

Usage example:

.. code-block:: python

    # Create a custom DenseCPT in the InputMaster dense_cpt 
    # InlineSection.
    input_master.dense_cpt["start"] = \
        DenseCPT([[0.7, 0.3], [0.8, 0.2]])
    # Create a DenseCPT with specified diagonal value and 
    # uniform other values
    input_master.dense_cpt["transition"] = \
        DenseCPT.uniform_from_shape(2, 2, self_transition = 0.6)


.. py:class:: DeterministicCPT

    A deterministic conditional probability table (CPT) described using an 
    existing decision tree with a specialized string method for writing to 
    the parameter file.

    .. py:attribute:: cardinality_parents
        :type: tuple[int]

        A tuple of integers describing the cardinality (number of states) for
        the parent variables. If it is empty, there are no parent variables.

    .. py:attribute:: cardinality
        :type: int

        The cardinality of this variable.

    .. py:attribute:: dt
        :type: str

        The name of the decision tree representing this deterministic CPT.

    .. py:method:: __init__(self, cardinality_parents, cardinality, dt)
        
        Creates a :py:class:`DeterministicCPT` with the provided attributes.

        :param cardinality_parents: The cardinality of parent variables
        :type cardinality_parents: tuple[int] or tuple
        :param cardinality: The cardinality of this variable
        :type cardinality: int
        :param dt: Name of an existing decision tree 
        :type dt: str 


Internal Classes
================

These classes are internal to the operation of the GMTK API, so the user
should not need to define or interact with these. However, they are documented
here for any developers interested in expanding or customizing the GMTK API.

Section Classes
---------------

Classes to store multiple objects that form one section of the parameter file.
These are used within the ``InputMaster`` object.

.. py:class:: InlineSection

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file. 

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user. 

.. py:class:: InlineMCSection

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file.

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user. 

    .. py:attribute:: mean
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`Mean` objects. 
        The value of ``mean`` parameters in :py:class:`MC` objects should be 
        keys in this object. 

    .. py:attribute:: covar
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`Covar` objects. 
        The value of ``covar`` parameters in :py:class:`MC` objects should be 
        keys in this object.  

.. py:class:: InlineMXSection

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file.

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user. 

    .. py:attribute:: dpmf
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`DPMF` objects. 
        The value of ``dpmf`` parameters in :py:class:`MX` objects should be 
        keys in this object. 


Abstract Parameter Classes
--------------------------

Abstract superclasses of the concrete Parameter classes described above. 

.. py:class:: Array

    An abstract class for array-like data, which inherits from Numpy's ``ndarray`` class.

    The ``__new__`` method (called on creating a new member of the class) is overwritten.
    It verifies the provided arguments have type ``int``, ``float``, or ``ndarray`` before 
    creating a class object containing those arguments. It also accepts the optional
    argument ``keep_shape``, which defaults to false. If 0-dimensional data (a single value)
    is provided and ``keep_shape`` is true, the created object will have 0 dimensions. 
    Otherwise, the created object will have 1 dimension and that value as the only item.

.. py:class:: OneLineKind
    
    An abstract class which is the parent for Array-like GMTK parameter 
    classes which have a one-line string representation, such as 
    :py:class:`Mean`, :py:class:`Covar`, and :py:class:`DPMF`.

    As a child of :py:class:`Array`, it behaves like a Numpy ``ndarray`` for data storage.
    However, when written to the input master file, its header and contents are printed 
    as a single line.

.. py:class:: Section

    A type-enforced dictionary with an additional string method for writing 
    to the parameter file. 

    .. py:attribute:: kind
        :type: str or None
        :value: None

        A string denoting the type which can be values in this 
        object. If not given, it is set by the first item. This should not be 
        changed by user.

    .. py:method:: __init__(self, kind)

        Create a Section object with the specified kind.

        :param kind: The kind for this section. All dictionary values in the section must have this string as their ``kind`` attribute.
        :type kind: str or None

