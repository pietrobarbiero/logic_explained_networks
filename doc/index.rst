Welcome to Deep Logic
======================


Deep Logic is a python package providing a set of utilities to
build deep learning models that are explainable by design.

This library provides APIs to:

* prune a standard model to get deep logic model
* extract logical formulas explaining network predictions
* validate the input data, the model architecture, and the pruning strategy

Quick start
-----------

You can install Deep Logic along with all its dependencies from
`PyPI <https://pypi.org/project/deeplogic/>`__:

.. code:: bash

    $ pip install -r requirements.txt deeplogic

Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pietrobarbiero/deeplogic>`__.


.. toctree::
    :caption: User Guide
    :maxdepth: 2

    user_guide/installation
    user_guide/tutorial
    user_guide/contributing
    user_guide/running_tests

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    modules/fol
    modules/prune
    modules/utils


.. toctree::
    :caption: Copyright
    :maxdepth: 1

    user_guide/authors
    user_guide/licence


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`