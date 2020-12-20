Welcome to Deep Logic
======================


|Build|
|Coverage|

|Docs|
|Dependendencies|

|PyPI license|
|PyPI-version|





.. |Build| image:: https://img.shields.io/travis/pietrobarbiero/deep-logic?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.org/pietrobarbiero/deep-logic

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/pietrobarbiero/deep-logic?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/pietrobarbiero/deep-logic

.. |Docs| image:: https://img.shields.io/readthedocs/deep-logic/latest?style=for-the-badge
    :alt: Read the Docs (version)
    :target: https://deep-logic.readthedocs.io/en/latest/

.. |Dependendencies| image:: https://img.shields.io/requires/github/pietrobarbiero/deep-logic?style=for-the-badge
    :alt: Requires.io
    :target: https://requires.io/github/pietrobarbiero/deep-logic/requirements/?branch=master

.. |Repo size| image:: https://img.shields.io/github/repo-size/pietrobarbiero/deep-logic?style=for-the-badge
    :alt: GitHub repo size
    :target: https://github.com/pietrobarbiero/deep-logic

.. |PyPI download total| image:: https://img.shields.io/pypi/dm/deep-logic?label=downloads&style=for-the-badge
    :alt: PyPI - Downloads
    :target: https://pypi.python.org/pypi/deep-logic/

.. |Open issues| image:: https://img.shields.io/github/issues/pietrobarbiero/deep-logic?style=for-the-badge
    :alt: GitHub issues
    :target: https://github.com/pietrobarbiero/deep-logic

.. |PyPI license| image:: https://img.shields.io/pypi/l/deep-logic.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/deeplogic/

.. |Followers| image:: https://img.shields.io/github/followers/pietrobarbiero?style=social
    :alt: GitHub followers
    :target: https://github.com/pietrobarbiero/deep-logic

.. |Stars| image:: https://img.shields.io/github/stars/pietrobarbiero/deep-logic?style=social
    :alt: GitHub stars
    :target: https://github.com/pietrobarbiero/deep-logic

.. |PyPI-version| image:: https://img.shields.io/pypi/v/deep-logic?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/deeplogic/

.. |Contributors| image:: https://img.shields.io/github/contributors/pietrobarbiero/deep-logic?style=for-the-badge
    :alt: GitHub contributors
    :target: https://github.com/pietrobarbiero/deep-logic

.. |Language| image:: https://img.shields.io/github/languages/top/pietrobarbiero/deep-logic?style=for-the-badge
    :alt: GitHub top language
    :target: https://github.com/pietrobarbiero/deep-logic

.. |Maintenance| image:: https://img.shields.io/maintenance/yes/2019?style=for-the-badge
    :alt: Maintenance
    :target: https://github.com/pietrobarbiero/deep-logic


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


Example
-----------

First of all we need to import some useful libraries:

.. code:: python

    import torch
    import numpy as np
    import deep_logic as dl

In most cases it is recommended to fix the random seed for
reproducibility:

.. code:: python

    torch.manual_seed(0)
    np.random.seed(0)

For this simple experiment, let's set up a simple toy problem
as the XOR problem:

.. code:: python

    x = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1], ], dtype=torch.float)
    y = torch.tensor([0, 1, 1, 0],
        dtype=torch.float).unsqueeze(1)

We can instantiate a simple feed-forward neural network with 2 layers:

.. code:: python

    layers = [
        torch.nn.Linear(2, 4),
        torch.nn.Sigmoid(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid()
    ]
    model = torch.nn.Sequential(*layers)

Before training the network, we should validate the input data and the
network architecture. The requirements are the following:

* all the input features should be in $[0,1]$;
* all the activation functions should be sigmoids.

.. code:: python

    dl.validate_data(x)
    dl.validate_network(model)

We can now train the network pruning weights with the
lowest absolute values after 500 epochs:

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(1000):
        # forward pass
        optimizer.zero_grad()
        y_pred = model(x)
        # Compute Loss
        loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
        # backward pass
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            y_pred_d = (y_pred > 0.5)
            accuracy = (y_pred_d.eq(y).sum(dim=1) == y.size(1)).sum().item() / y.size(0)
            print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

        # pruning
        if epoch > 500:
            model = dl.prune_equal_fanin(model, 2)

Once trained the ``fol`` package can be used to generate first-order
logic explanations of the predictions:

.. code:: python

    # generate explanations
    weights, biases = dl.collect_parameters(model)
    f = dl.fol.generate_fol_explanations(weights, biases)[0]
    print(f'Explanation: {f}')

For this problem the generated explanation is ``(f1 & ~f2) | (f2 & ~f1)``
which corresponds to ``f1 XOR f2``.

Theory
--------
Theoretical foundations can be found in the following papers.

Learning of constraints::

    @inproceedings{ciravegna2020constraint,
      title={A Constraint-Based Approach to Learning and Explanation.},
      author={Ciravegna, Gabriele and Giannini, Francesco and Melacci, Stefano and Maggini, Marco and Gori, Marco},
      booktitle={AAAI},
      pages={3658--3665},
      year={2020}
    }

Learning with constraints::

    @inproceedings{marra2019lyrics,
      title={LYRICS: A General Interface Layer to Integrate Logic Inference and Deep Learning},
      author={Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and Gori, Marco},
      booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
      pages={283--298},
      year={2019},
      organization={Springer}
    }

Constraints theory in machine learning::

    @book{gori2017machine,
      title={Machine Learning: A constraint-based approach},
      author={Gori, Marco},
      year={2017},
      publisher={Morgan Kaufmann}
    }


Authors
-------

`Pietro Barbiero <http://www.pietrobarbiero.eu/>`__, Gabriele Ciravegna, and Dobrik Georgiev.

Licence
-------

Copyright 2020 Pietro Barbiero.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.