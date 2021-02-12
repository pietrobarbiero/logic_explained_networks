Welcome to Deep Logic
-----------------------


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
   :target: https://pypi.python.org/pypi/deep-logic/

.. |Followers| image:: https://img.shields.io/github/followers/pietrobarbiero?style=social
    :alt: GitHub followers
    :target: https://github.com/pietrobarbiero/deep-logic

.. |Stars| image:: https://img.shields.io/github/stars/pietrobarbiero/deep-logic?style=social
    :alt: GitHub stars
    :target: https://github.com/pietrobarbiero/deep-logic

.. |PyPI-version| image:: https://img.shields.io/pypi/v/deep-logic?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/deep-logic/

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

This library provides APIs to get first-order logic explanations
from neural networks.

Quick start
-----------

You can install Deep Logic along with all its dependencies from
`PyPI <https://pypi.org/project/deep-logic/>`__:

.. code:: bash

    pip install -r requirements.txt deep-logic


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

    set_seed(0)

For this simple experiment, let's set up a simple toy problem
as the XOR problem (plus 2 dummy features):

.. code:: python

    x_train = torch.tensor([
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
    ], dtype=torch.float)
    y_train = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1)
    xnp = x_train.detach().numpy()
    ynp = y_train.detach().numpy().ravel()

We can instantiate a simple feed-forward neural network with 3 layers:

.. code:: python

    layers = [
        torch.nn.Linear(x_train.size(1), 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid(),
    ]
    model = torch.nn.Sequential(*layers)

Before training the network, we should validate the input data.
The only requirement is the following for all the input features to be in ``[0,1]``.

.. code:: python

    dl.validate_data(x_train)

We can now train the network:

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    need_pruning = True
    for epoch in range(1000):
        # forward pass
        optimizer.zero_grad()
        y_pred = model(x_train)

        # Compute Loss
        loss = torch.nn.functional.binary_crossentropy_loss(y_pred, y_train)
        # A bit of L1 regularization will encourage sparsity
        for module in model.children():
            if isinstance(module, torch.nn.Linear):
                loss += 0.001 * torch.norm(module.weight, 1)

        # We can use sparsity to prune dummy features
        if epoch > 500 and need_pruning:
            dl.utils.relu_nn.prune_features(model, n_classes)
            need_pruning = False


        # backward pass
        loss.backward()
        optimizer.step()

        # compute accuracy
        if epoch % 100 == 0:
            y_pred_d = (y_pred > 0.5)
            accuracy = (y_pred_d.eq(y_train).sum(dim=1) == y_train.size(1)).sum().item() / y_train.size(0)
            print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')


Once trained we can extract first-order logic formulas describing
local explanations of the prediction for a specific input by looking
at the reduced model:

.. code:: python

    explanation = dl.logic.explain_local(model, x_train, y_train, x_sample=x[1],
                                         method='pruning', target_class=1,
                                         concept_names=['f1', 'f2', 'f3', 'f4'])
    print(explanation)

The local explanation will be a given in terms of conjunctions
of input features which are locally relevant (the dummy features
will be discarded thanks to pruning).
For this specific input, the explanation would be
``~f1 AND f2``.

Finally the ``fol`` package can be used to generate global
explanations of the predictions for a specific class:

.. code:: python


    global_explanation, _, _ = dl.logic.relu_nn.combine_local_explanations(model, x_train,
                                                                          y_train.squeeze(),
                                                                          target_class=1,
                                                                          method='pruning')
    accuracy, _ = dl.logic.base.test_explanation(global_explanation, target_class=1, x_train, y_train)
    explanation = dl.logic.base.replace_names(global_explanation, concept_names=['f1', 'f2', 'f3', 'f4'])
    print(f'Accuracy when using the formula {explanation}: {accuracy:.4f}')


The global explanation is given in a disjunctive normal form
for a specified class.
For this problem the generated explanation for class ``y=1`` is
``(f1 AND ~f2) OR (f2  AND ~f1)``
which corresponds to ``f1 XOR f2``
(i.e. the `exclusive OR` function).

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

* `Pietro Barbiero <http://www.pietrobarbiero.eu/>`__, University ofCambridge, UK.
* Francesco Giannini, University of Florence, IT.
* Gabriele Ciravegna, University of Florence, IT.
* Dobrik Georgiev, University of Cambridge, UK.


Licence
-------

Copyright 2020 Pietro Barbiero, Francesco Giannini, Gabriele Ciravegna, and Dobrik Georgiev.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.