=============
Welcome to the Logic Explained Networks (LENs) repo
=============


Logic Explained Network is a python repository providing a set of utilities and modules to
build deep learning models that are explainable by design.

This library provides both already implemented LENs classes and APIs classes to get First-Order Logic (FOL) explanations from neural networks.

Paper
=============

The theory behind this library has been published in the `Logic Explained Networks <https://www.sciencedirect.com/science/article/pii/S000437022200162X>`__
paper, published in the Artificial Intelligence journal.

A publicly available version can be downloaded from `ArXiv <https://arxiv.org/abs/2108.05149>`__


Structure of the repository
=============

.. code::

    LENs
    ├── data
    ├── examples
    │     ├── example.py
    |     └── api_examples
    ├── experiments
    ├── lens
    │     ├── logic
    |     ├── models
    │         ├── mu_nn
              ├── psi_nn
              └── relu_nn
    |     └── utils
    ├── tests
    │   ├── test_models
        ├── test_logic_base
        └── test_concept_extractors

The most important folder is of course `lens.models` where you can find all the proposed Logic Explained Networks: the $\psi$ network

Example
=============
Import LENs
------------

.. code:: python

   import lens
   import torch

Create train, validation and test datasets
------------

Let’s create a XOR-like datasets with 2 redundant features (the 3rd and
4th).

.. code:: python

   lens.utils.base.set_seed(0)
   x = torch.rand([100, 4])
   y = (x[:, 0] > 0.5) & (x[:, 1] < 0.5) | \
       (x[:, 0] < 0.5) & (x[:, 1] > 0.5)

   data = torch.utils.data.TensorDataset(x, y)

   train_data, val_data, test_data = torch.utils.data.random_split(data, [80, 10, 10])
   x_train, y_train = data[train_data.indices]
   x_val, y_val = data[val_data.indices]
   x_test, y_test = data[test_data.indices]

Instantiate a GeneralNN
------------

.. code:: python

   model = lens.models.XMuNN(n_classes=2, n_features=4,
                                  hidden_neurons=[10], loss=torch.nn.CrossEntropyLoss())

Train the model
------------

.. code:: python

   model.fit(train_data, val_data, epochs=100, l_r=0.1)

   ## get accuracy on test samples
   test_acc = model.evaluate(test_data)
   print("Test accuracy:", test_acc)

::

   Epoch: 1/100, Loss: 0.733, Tr_acc: 46.88, Val_acc: 53.00, best_e: -1
   Epoch: 2/100, Loss: 0.733, Tr_acc: 53.12, Val_acc: 53.00, best_e: -1
   Epoch: 3/100, Loss: 0.706, Tr_acc: 53.12, Val_acc: 65.00, best_e: -1
   Epoch: 4/100, Loss: 0.696, Tr_acc: 58.50, Val_acc: 46.00, best_e: -1
   Epoch: 5/100, Loss: 0.700, Tr_acc: 45.25, Val_acc: 43.00, best_e: -1
   Epoch: 6/100, Loss: 0.698, Tr_acc: 42.25, Val_acc: 64.00, best_e: -1
   Epoch: 7/100, Loss: 0.694, Tr_acc: 62.50, Val_acc: 53.00, best_e: -1
   Epoch: 8/100, Loss: 0.694, Tr_acc: 53.12, Val_acc: 55.00, best_e: -1
   Epoch: 9/100, Loss: 0.691, Tr_acc: 55.88, Val_acc: 65.00, best_e: -1
   Epoch: 10/100, Loss: 0.687, Tr_acc: 62.25, Val_acc: 63.00, best_e: -1
   Epoch: 11/100, Loss: 0.682, Tr_acc: 63.38, Val_acc: 58.00, best_e: -1
   Epoch: 12/100, Loss: 0.678, Tr_acc: 62.50, Val_acc: 61.00, best_e: -1
   Epoch: 13/100, Loss: 0.673, Tr_acc: 66.12, Val_acc: 70.00, best_e: -1
   Epoch: 14/100, Loss: 0.664, Tr_acc: 71.50, Val_acc: 72.00, best_e: -1
   Epoch: 15/100, Loss: 0.654, Tr_acc: 75.00, Val_acc: 74.00, best_e: -1
   Epoch: 16/100, Loss: 0.643, Tr_acc: 74.75, Val_acc: 74.00, best_e: -1
   Epoch: 17/100, Loss: 0.628, Tr_acc: 75.12, Val_acc: 73.00, best_e: -1
   Epoch: 18/100, Loss: 0.612, Tr_acc: 74.75, Val_acc: 73.00, best_e: -1
   Epoch: 19/100, Loss: 0.594, Tr_acc: 78.38, Val_acc: 77.00, best_e: -1
   Epoch: 20/100, Loss: 0.574, Tr_acc: 81.38, Val_acc: 81.00, best_e: -1
   Epoch: 21/100, Loss: 0.553, Tr_acc: 81.88, Val_acc: 82.00, best_e: -1
   Epoch: 22/100, Loss: 0.529, Tr_acc: 82.50, Val_acc: 81.00, best_e: -1
   Epoch: 23/100, Loss: 0.504, Tr_acc: 83.12, Val_acc: 78.00, best_e: -1
   Epoch: 24/100, Loss: 0.481, Tr_acc: 81.88, Val_acc: 81.00, best_e: -1
   Epoch: 25/100, Loss: 0.456, Tr_acc: 83.50, Val_acc: 83.00, best_e: -1
   Epoch: 26/100, Loss: 0.430, Tr_acc: 84.88, Val_acc: 87.00, best_e: -1
   Epoch: 27/100, Loss: 0.407, Tr_acc: 85.88, Val_acc: 89.00, best_e: -1
   Epoch: 28/100, Loss: 0.384, Tr_acc: 86.50, Val_acc: 88.00, best_e: -1
   Epoch: 29/100, Loss: 0.362, Tr_acc: 86.62, Val_acc: 88.00, best_e: -1
   Epoch: 30/100, Loss: 0.341, Tr_acc: 86.75, Val_acc: 89.00, best_e: -1
   Epoch: 31/100, Loss: 0.321, Tr_acc: 88.25, Val_acc: 89.00, best_e: -1
   Epoch: 32/100, Loss: 0.304, Tr_acc: 88.62, Val_acc: 89.00, best_e: -1
   Epoch: 33/100, Loss: 0.287, Tr_acc: 88.75, Val_acc: 90.00, best_e: -1
   Epoch: 34/100, Loss: 0.271, Tr_acc: 89.75, Val_acc: 87.00, best_e: -1
   Epoch: 35/100, Loss: 0.259, Tr_acc: 90.12, Val_acc: 90.00, best_e: -1
   Epoch: 36/100, Loss: 0.249, Tr_acc: 90.75, Val_acc: 88.00, best_e: -1
   Epoch: 37/100, Loss: 0.235, Tr_acc: 91.12, Val_acc: 89.00, best_e: -1
   Epoch: 38/100, Loss: 0.222, Tr_acc: 92.00, Val_acc: 89.00, best_e: -1
   Epoch: 39/100, Loss: 0.214, Tr_acc: 92.00, Val_acc: 90.00, best_e: -1
   Epoch: 40/100, Loss: 0.204, Tr_acc: 92.50, Val_acc: 91.00, best_e: -1
   Epoch: 41/100, Loss: 0.193, Tr_acc: 92.62, Val_acc: 92.00, best_e: -1
   Epoch: 42/100, Loss: 0.187, Tr_acc: 93.25, Val_acc: 91.00, best_e: -1
   Epoch: 43/100, Loss: 0.178, Tr_acc: 94.50, Val_acc: 91.00, best_e: -1
   Epoch: 44/100, Loss: 0.169, Tr_acc: 94.12, Val_acc: 92.00, best_e: -1
   Epoch: 45/100, Loss: 0.164, Tr_acc: 94.62, Val_acc: 91.00, best_e: -1
   Epoch: 46/100, Loss: 0.156, Tr_acc: 95.88, Val_acc: 92.00, best_e: -1
   Epoch: 47/100, Loss: 0.149, Tr_acc: 96.00, Val_acc: 92.00, best_e: -1
   Epoch: 48/100, Loss: 0.144, Tr_acc: 97.12, Val_acc: 93.00, best_e: -1
   Epoch: 49/100, Loss: 0.139, Tr_acc: 97.12, Val_acc: 93.00, best_e: -1
   Pruned 2/4 features
   Pruned 2/4 features
   Pruned features
   Epoch: 50/100, Loss: 0.133, Tr_acc: 97.62, Val_acc: 93.00, best_e: -1
   Epoch: 51/100, Loss: 0.140, Tr_acc: 94.62, Val_acc: 78.00, best_e: 51
   Epoch: 52/100, Loss: 0.363, Tr_acc: 81.88, Val_acc: 90.00, best_e: 52
   Epoch: 53/100, Loss: 0.146, Tr_acc: 95.62, Val_acc: 92.00, best_e: 53
   Epoch: 54/100, Loss: 0.165, Tr_acc: 92.00, Val_acc: 88.00, best_e: 53
   Epoch: 55/100, Loss: 0.237, Tr_acc: 86.75, Val_acc: 89.00, best_e: 53
   Epoch: 56/100, Loss: 0.214, Tr_acc: 88.12, Val_acc: 93.00, best_e: 56
   Epoch: 57/100, Loss: 0.152, Tr_acc: 92.50, Val_acc: 92.00, best_e: 56
   Epoch: 58/100, Loss: 0.126, Tr_acc: 97.62, Val_acc: 91.00, best_e: 56
   Epoch: 59/100, Loss: 0.149, Tr_acc: 95.38, Val_acc: 89.00, best_e: 56
   Epoch: 60/100, Loss: 0.177, Tr_acc: 92.00, Val_acc: 89.00, best_e: 56
   Epoch: 61/100, Loss: 0.172, Tr_acc: 92.62, Val_acc: 92.00, best_e: 56
   Epoch: 62/100, Loss: 0.144, Tr_acc: 95.25, Val_acc: 93.00, best_e: 62
   Epoch: 63/100, Loss: 0.124, Tr_acc: 97.88, Val_acc: 96.00, best_e: 63
   Epoch: 64/100, Loss: 0.126, Tr_acc: 96.50, Val_acc: 94.00, best_e: 63
   Epoch: 65/100, Loss: 0.142, Tr_acc: 93.62, Val_acc: 92.00, best_e: 63
   Epoch: 66/100, Loss: 0.150, Tr_acc: 92.88, Val_acc: 94.00, best_e: 63
   Epoch: 67/100, Loss: 0.141, Tr_acc: 93.50, Val_acc: 94.00, best_e: 63
   Epoch: 68/100, Loss: 0.126, Tr_acc: 95.88, Val_acc: 97.00, best_e: 68
   Epoch: 69/100, Loss: 0.117, Tr_acc: 98.62, Val_acc: 93.00, best_e: 68
   Epoch: 70/100, Loss: 0.121, Tr_acc: 98.00, Val_acc: 92.00, best_e: 68
   Epoch: 71/100, Loss: 0.130, Tr_acc: 96.12, Val_acc: 92.00, best_e: 68
   Epoch: 72/100, Loss: 0.131, Tr_acc: 95.88, Val_acc: 92.00, best_e: 68
   Epoch: 73/100, Loss: 0.123, Tr_acc: 97.25, Val_acc: 95.00, best_e: 68
   Epoch: 74/100, Loss: 0.114, Tr_acc: 98.25, Val_acc: 98.00, best_e: 74
   Epoch: 75/100, Loss: 0.113, Tr_acc: 98.38, Val_acc: 95.00, best_e: 74
   Epoch: 76/100, Loss: 0.117, Tr_acc: 96.38, Val_acc: 94.00, best_e: 74
   Epoch: 77/100, Loss: 0.120, Tr_acc: 95.75, Val_acc: 95.00, best_e: 74
   Epoch: 78/100, Loss: 0.118, Tr_acc: 96.25, Val_acc: 97.00, best_e: 74
   Epoch: 79/100, Loss: 0.112, Tr_acc: 97.75, Val_acc: 99.00, best_e: 79
   Epoch: 80/100, Loss: 0.109, Tr_acc: 99.12, Val_acc: 97.00, best_e: 79
   Epoch: 81/100, Loss: 0.110, Tr_acc: 98.38, Val_acc: 94.00, best_e: 79
   Epoch: 82/100, Loss: 0.113, Tr_acc: 97.88, Val_acc: 96.00, best_e: 79
   Epoch: 83/100, Loss: 0.112, Tr_acc: 97.75, Val_acc: 97.00, best_e: 79
   Epoch: 84/100, Loss: 0.108, Tr_acc: 98.50, Val_acc: 99.00, best_e: 84
   Epoch: 85/100, Loss: 0.105, Tr_acc: 99.25, Val_acc: 98.00, best_e: 84
   Epoch: 86/100, Loss: 0.106, Tr_acc: 98.25, Val_acc: 96.00, best_e: 84
   Epoch: 87/100, Loss: 0.107, Tr_acc: 97.75, Val_acc: 97.00, best_e: 84
   Epoch: 88/100, Loss: 0.106, Tr_acc: 97.75, Val_acc: 98.00, best_e: 84
   Epoch: 89/100, Loss: 0.104, Tr_acc: 98.25, Val_acc: 99.00, best_e: 89
   Epoch: 90/100, Loss: 0.102, Tr_acc: 99.12, Val_acc: 96.00, best_e: 89
   Epoch: 91/100, Loss: 0.102, Tr_acc: 98.50, Val_acc: 97.00, best_e: 89
   Epoch: 92/100, Loss: 0.103, Tr_acc: 98.38, Val_acc: 97.00, best_e: 89
   Epoch: 93/100, Loss: 0.102, Tr_acc: 98.50, Val_acc: 97.00, best_e: 89
   Epoch: 94/100, Loss: 0.100, Tr_acc: 99.00, Val_acc: 98.00, best_e: 89
   Epoch: 95/100, Loss: 0.099, Tr_acc: 98.88, Val_acc: 98.00, best_e: 89
   Epoch: 96/100, Loss: 0.099, Tr_acc: 98.12, Val_acc: 98.00, best_e: 89
   Epoch: 97/100, Loss: 0.099, Tr_acc: 98.12, Val_acc: 98.00, best_e: 89
   Epoch: 98/100, Loss: 0.098, Tr_acc: 98.38, Val_acc: 98.00, best_e: 89
   Epoch: 99/100, Loss: 0.097, Tr_acc: 98.88, Val_acc: 97.00, best_e: 89
   Epoch: 100/100, Loss: 0.096, Tr_acc: 98.75, Val_acc: 97.00, best_e: 89
   Test accuracy: 99.0

Extract and evaluate global explanation
---------------------------------------

.. code:: python

   ## get first order logic explanations for a specific target class
   target_class = 1
   concept_names = ['x1', 'x2', 'x3', 'x4']
   formula = model.get_global_explanation(x_train, y_train, target_class,
                                          top_k_explanations=2, concept_names=concept_names)
   print(f"{formula} <-> f{target_class}")

   ## compute explanation accuracy
   exp_accuracy, _ = lens.logic.test_explanation(formula, target_class, x_test, y_test,
                                                 concept_names=concept_names)
   print("Logic Test Accuracy:", exp_accuracy)

::

   (x1 & ~x2) | (x2 & ~x1) <-> f1
   Logic Test Accuracy: 100.0

Plot decision boundaries and explanations
-----------------------------------------

.. code:: python

   import numpy as np
   import matplotlib.pyplot as plt


   def plot_decision_bundaries(model, x, h=0.1, cmap='BrBG'):
       x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
       x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
       xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                              np.arange(x2_min, x2_max, h))
       xx = torch.FloatTensor(np.c_[xx1.ravel(), xx2.ravel(), xx1.ravel(), xx2.ravel()])
       Z = model(xx).argmax(dim=1).detach().numpy()
       Z = Z.reshape(xx1.shape)
       plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
       return

.. code:: python

   x = torch.as_tensor([[0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0]])
   y = torch.as_tensor([0, 1, 1, 0])
   cmap = 'BrBG'
   plt.figure(figsize=[8, 8])
   for sample_id, (xin, yin) in enumerate(zip(x, y)):
       output = model(xin.unsqueeze(dim=0))
       explanation = model.get_local_explanation(x_train, y_train, xin, yin,
                                                 concept_names=concept_names)

       plt.subplot(2, 2, sample_id + 1)
       plt.title(f'INPUT={xin[:2].detach().numpy()} - OUTPUT={output.argmax(dim=1).detach().numpy()} '
                 f'\n Explanation: {explanation} -> f{output.argmax()}')
       plot_decision_bundaries(model, x, h=0.01)
       plt.scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), c=y.detach().numpy(), cmap=cmap)
       plt.scatter(xin[0], xin[1], c='k', marker='x', s=100, cmap=cmap)
       c = plt.Circle((xin[0], xin[1]), radius=0.2, edgecolor='k', fill=False, linestyle='--')
       plt.gca().add_artist(c)
       plt.xlim([-0.5, 1.5])
       plt.ylim([-0.5, 1.5])
   plt.tight_layout()
   plt.show()

.. figure:: output_12_0.png
   :alt: png


Citation and theory
=============
To cite the Logic Explained Network paper use the following bibtex::

    @article{DBLP:journals/corr/abs-2108-05149,
        author    = {Gabriele Ciravegna and
                   Pietro Barbiero and
                   Francesco Giannini and
                   Marco Gori and
                   Pietro Li{\'{o}} and
                   Marco Maggini and
                   Stefano Melacci},
        title     = {Logic Explained Networks},
        journal   = {CoRR},
        volume    = {abs/2108.05149},
        year      = {2021},
        url       = {https://arxiv.org/abs/2108.05149},
        eprinttype = {arXiv},
        eprint    = {2108.05149},
        timestamp = {Wed, 18 Aug 2021 19:45:42 +0200},
        biburl    = {https://dblp.org/rec/journals/corr/abs-2108-05149.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }

Theoretical foundations of this work can be found in the following papers:

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

* `Gabriele Ciravegna <https://team.inria.fr/maasai/gabriele-ciravegna/>`__, Université Cote d'Azur, FR.
* `Pietro Barbiero <http://www.pietrobarbiero.eu/>`__, University of Cambridge, UK.
* `Francesco Giannini <http://sailab.diism.unisi.it/people/francesco-giannini/>`__, University of Siena, IT.


Licence
-------

Copyright 2022 Gabriele Ciravegna, Pietro Barbiero, Francesco Giannini.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.