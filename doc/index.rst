Welcome to Deep Logic
======================


Deep Logic is a python package providing a set of utilities to
build deep learning models that are explainable by design.

This library provides APIs to get first-order logic explanations from:

* ReLU networks (:doc:`../user_guide/tutorial_relu`);
* :math:`\psi`-networks, i.e. neural networks with sigmoid activations (:doc:`../user_guide/tutorial_psi`).

Quick start
-----------

You can install Deep Logic along with all its dependencies from
`PyPI <https://pypi.org/project/deep-logic/>`__:

.. code:: bash

    $ pip install -r requirements.txt deep-logic


Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/pietrobarbiero/deep-logic>`__.


.. toctree::
    :caption: User Guide
    :maxdepth: 2

    user_guide/installation
    user_guide/tutorial_relu
    user_guide/tutorial_psi
    user_guide/contributing
    user_guide/running_tests

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    modules/relunn
    modules/fol_relu
    modules/prune
    modules/fol_psi
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