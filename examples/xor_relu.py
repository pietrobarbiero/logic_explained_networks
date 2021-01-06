import torch
import numpy as np
import matplotlib.pyplot as plt
from sympy import simplify_logic

import deep_logic as dl

torch.manual_seed(0)
np.random.seed(0)

# XOR problem
x_train = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=torch.float)
y_train = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1)
xnp = x_train.detach().numpy()
ynp = y_train.detach().numpy().ravel()

layers = [
    torch.nn.Linear(x_train.size(1), 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
    torch.nn.Sigmoid(),
]
model = torch.nn.Sequential(*layers)

dl.validate_network(model, model_type='relu')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(1000):
    # forward pass
    optimizer.zero_grad()
    y_pred = model(x_train)
    # Compute Loss
    loss = torch.nn.functional.mse_loss(y_pred, y_train)

    for module in model.children():
        if isinstance(module, torch.nn.Linear):
            loss += 0.001 * torch.norm(module.weight, 1)

    # backward pass
    loss.backward()
    optimizer.step()

    # compute accuracy
    if epoch % 100 == 0:
        y_pred_d = (y_pred > 0.5)
        accuracy = (y_pred_d.eq(y_train).sum(dim=1) == y_train.size(1)).sum().item() / y_train.size(0)
        print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

# Decision boundaries

def plot_decision_bundaries(model, x, h=0.1, cmap='BrBG'):
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))
    xx = torch.FloatTensor(np.c_[xx1.ravel(), xx2.ravel()])
    Z = model(xx).detach().numpy()
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    return

## Local decision boundaries

xin = torch.tensor([0.3, 0.95])
model_reduced = dl.get_reduced_model(model, xin)
output = model_reduced(xin)
explanation = dl.fol.generate_local_explanations(model_reduced, xin)

plt.figure(figsize=[8, 4])
plt.subplot(121)
plt.title('True decision boundary')
plot_decision_bundaries(model, x_train, h=0.01)
plt.scatter(xin[0], xin[1], c='k', marker='x', s=100)
c = plt.Circle((xin[0], xin[1]), radius=0.2, edgecolor='k', fill=False, linestyle='--')
plt.gca().add_artist(c)
plt.scatter(xnp[:, 0], xnp[:, 1], c=ynp, cmap='BrBG')
plt.xlabel('f0')
plt.ylabel('f1')
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
plt.subplot(122)
plt.title(f'IN={xin.detach().numpy()} - OUT={output.detach().numpy()}\nExplanation: {explanation}')
plot_decision_bundaries(model_reduced, x_train)
plt.scatter(xin[0], xin[1], c='k', marker='x', s=100)
c = plt.Circle((xin[0], xin[1]), radius=0.2, edgecolor='k', fill=False, linestyle='--')
plt.gca().add_artist(c)
plt.scatter(xnp[:, 0], xnp[:, 1], c=ynp, cmap='BrBG')
plt.xlabel('f0')
plt.ylabel('f1')
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
plt.savefig('decision_boundaries.png')
plt.show()

global_explanation = dl.fol.combine_local_explanations(model, x_train, y_train)

simplified_explanation = simplify_logic(global_explanation, 'dnf')
simplified_explanation
