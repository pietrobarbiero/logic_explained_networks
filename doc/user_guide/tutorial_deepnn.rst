FOL explanations of deep networks
=======================================


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
            dl.utils.relunn.prune_features(model, n_classes)
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


    global_explanation, _, _ = dl.logic.relunn.combine_local_explanations(model, x_train,
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
