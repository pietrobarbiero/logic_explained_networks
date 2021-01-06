FOL explanations of :math:`\psi`-networks
==========================================

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

* all the input features should be in :math:`[0,1]`;
* all the activation functions should be sigmoids.

.. code:: python

    dl.validate_data(x)
    dl.validate_network(model, 'psi')

We can now train the network pruning weights with the
lowest absolute values after 500 epochs:

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    need_pruning = True
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
        if epoch > 500 and need_pruning:
            model = dl.prune_equal_fanin(model, 2)
            need_pruning = False

Once trained the ``fol`` package can be used to generate first-order
logic explanations of the predictions:

.. code:: python

    # generate explanations
    weights, biases = dl.collect_parameters(model)
    f = dl.fol.generate_fol_explanations(weights, biases)[0]
    print(f'Explanation: {f}')

For this problem the generated explanation for class :math:`y=1` is
:math:`(f_1 \land \neg f_2) \lor (f_2  \land \neg f_1)`
which corresponds to :math:`f_1 \oplus f_2`
(i.e. the exclusive OR function).