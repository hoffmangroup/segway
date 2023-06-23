====================
Segway API Reference
====================

Description
===========

The Segway API provides a Python framework to design dynamic Bayesian models for custom 
tasks. Segway can perform training and inference using these user-defined models.

The Segway API is installed with Segway automatically. 

Workflow
========

The Segway API describes the model using two files:

  1. A structure file describing the graphical network's nodes and 
  edges. This should be written using the GMTK structure language. The GMTK 
  structure language is described in the 
  `GMTK Documentation <https://github.com/melodi-lab/gmtk/blob/master/documentation.pdf>`_.

  2. A Python file describing the initial parameter settings and Segway 
  commands for training and inference. The classes provided by the Segway API
  for writing this file are described below. It should contain 3 areas:

    1. Code defining an :py:class:`InputMaster` object and setting its 
    appropriate attributes to describe the model.

    2. A line saving the :py:class:`InputMaster` object to an input file, 
    which describes the parameters in GMTK structure format. 

    3. Code calling Segway to train the defined model and annotate using the 
    trained model. More information on running Segway is available in 
    :ref:`python-interface`.

A worked example applying the Segway API can be seen in the CNVway code.

.. todo: other section? flip sentence order? link?

InputMaster Class
=================

Central class storing all parameter information.

.. py:class:: InputMaster

    .. py:attribute:: name_collection
        :type: InlineSection
    
    Stores the names of hidden states (segmentation labels) in the model. 
    
    Behaves as a dictionary where keys are the hidden state name, which 
    can be referenced in the structure file, and each value should be set to 
    a list of state names or a :py:class:`NameCollection` object initialized 
    with state names. If a Python list is given, it is converted to a 
    :py:class:`NameCollection` object using :py:meth:`NameCollection.__init__`.

    .. py:attribute:: mean
        :type: InlineSection

    Stores the means of emission distributions for each hidden state. 
    
    Behaves as a dictionary where keys are names for later reference and each
    value should be set to the mean value or a :py:class:`Mean` object 
    initialized with the mean value. If a Python float or list of floats is 
    given, it is converted to a :py:class:`Mean` object using 
    :py:meth:`Mean.__init__`.

    .. py:attribute:: covar
        :type: InlineSection

    Stores the covariance of emission distributions for each hidden state.
    
    Behaves as a dictionary where keys are names for later reference and each
    value should be set to the covariance value or a :py:class:`Covar` object 
    initialized with the covariance value. If a Python float or list of floats is 
    given, it is converted to a :py:class:`Covar` object using 
    :py:meth:`Covar.__init__`.

    .. py:attribute:: dpmf
        :type: InlineSection

    Stores the Dense Probability Mass Function (DPMF), which can later be used
    to define Gaussian Mixture models. 
    
    Behaves as a dictionary where keys are names for later reference and each 
    value should be set to a :py:class:`DPMF` object. 

    .. py:attribute:: mc
        :type: InlineMCSection
        :value: InlineMCSection(mean = self.mean, covar = self.covar)

    Stores Gaussians acting as mixture components (MC) for a Gaussian mixture
    model on the emission distribution. 
    
    Behaves as a dictionary where keys are names for later reference and each 
    value should be set to an :py:class:`MC` object.

    .. py:attribute:: mx
        :type: InlineMXSection
    
    Store Gaussian mixture distributions constructed from above-defined mixture 
    components. 
    
    Behaves as a dictionary where keys are hidden state names (from 
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

Parameter Classes
=================

Class representing user-defined model parameters.

.. py:class:: NameCollection

    A list of names with a specialized string method for writing to the 
    parameter file.

    .. py:method:: __init__(self, *args)

        Create a :py:class:`NameCollection` object containing the provided names.

        :param args: List of names or names as multiple arguments.
        :type names: list[str] or str

.. py:class:: Mean

    A Numpy ``ndarray`` representing a distribution's mean, with a specialized 
    string method for writing to the parameter file. Supports monovariate 
    and multivariate distributions.

    .. py:method:: __init__(self, *args)

        Create a :py:class:`Mean` object storing the provided mean value or 
        vector.

        :param args: The mean value which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

    .. py:method:: get_dimension(self)

        Return the dimension of this object.

        :return: The dimension of the mean array
        :rtype: int

.. py:class:: Covar

    A Numpy ``ndarray`` representing a distribution's covariance, with a 
    specialized string method for writing to the parameter file. Supports 
    monovariate and multivariate distributions.

    .. py:method:: __init__(self, *args)

        Create a :py:class:`Covar` object storing the provided covariance 
        value or vector.

        :param args: The covariance value which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

    .. py:method:: get_dimension(self)

        Return the dimension of this object.

        :return: The dimension of the covariance array
        :rtype: int

.. py:class:: DPMF

    A Numpy ``ndarray`` representing a DPMF with a specialized string method 
    for writing to the parameter file. As it is intended for use in Gaussian 
    mixture models, it supports monovariate distributions only. 

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

    .. py:method:: get_length(self)

        Return the length of this object.

        :return: The length of the DPMF array, equal to the number of outcomes for the DPMF
        :rtype: int

.. py:class:: DiagGaussianMC

    A Gaussian distribution with diagonal covariance. Currently the only 
    concrete MC subclass which can be used as a mixture component. 

    .. py:attribute:: mean
        :type: Mean

        A :py:class:`Mean` object representing the mean of this Gaussian
    
    .. py:attribute:: covar
        :type: Covar
        
        A :py:class:`Covar` object representing the covariance vector along the 
        diagonal of the covariance matrix. 

    .. py:method:: __init__(self, mean, covar)

        Create a :py:class:`DiagGaussianMC` object with the specified mean 
        and covariance.

        :param mean: mean of the distribution
        :type mean: Mean
        :param covar: diagonal covariance vector of the distribution
        :type covar: Covar

.. py:class:: MX

    A Gaussian mixture model built from Gaussian mixture components.

    .. py:attribute:: dpmf
        :type: DPMF

        A :py:class:`DPMF` object representing the contribution of each 
        Gaussian mixture component to the mixture model.
    
    .. py:attribute:: components
        :type: str or list[str]

        Names of Gaussian components associated with the mixture model. 

    .. py:method:: __init__(self, dpmf, components)

        Create an :py:class:`MX` object with the mixture distribution and 
        components.

        :param dpmf: DPMF describing mixture weights.
        :type dpmf: DPMF
        :param components: Name or list of names of mixture components
        :type components: str or list[str]

.. py:class:: DenseCPT

    A Numpy ``ndarray`` representing a dense CPT with a specialized string method 
    for writing to the parameter file. Supports up to 3 dimensional tables.
    
    .. py:method:: __init__(self, *args)

        Create a :py:class:`DenseCPT` object storing the provided distribution.

        :param args: The probability distribution as an array of probabilties which is interpreted by the Numpy ``array`` constructor. 
        :type args: array_like

    .. py:classmethod:: uniform_from_shape(*shape, self=0.0)

        A class method for creating a :py:class:`DenseCPT` object with the provided 
        shape.
        If the table is 2 or 3 dimensional, the diagonal entries of the table 
        are set to the ``self`` parameter (default 0) and all other entries 
        are set to be uniform. 

        :param shape: Shape of Dense CPT table
        :type shape: Array_like or multiple arguments
        :param self: Value for diagonal entries in the table. Defaults to 0.0
        :type self: float

.. py:class:: DeterministicCPT

    A deterministic CPT described using an existing decision tree with a 
    specialized string method for writing to the parameter file.

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

Section Classes
===============

Classes to store multiple objects that form one section of the parameter file. 

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

    .. py:attribute:: cover
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

    .. py:attribute:: mc
        :type: InlineSection

        An :py:class:`InlineSection` object storing :py:class:`MC` objects. 
        The value of ``covar`` parameters in :py:class:`MX` objects should be 
        keys in this object.  

