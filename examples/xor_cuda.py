import os
import torch
import numpy as np
from deep_logic import validate_network, prune_equal_fanin, collect_parameters
from deep_logic import fol

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)
    np.random.seed(0)

    # XOR problem
    x = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1], ], dtype=torch.float).to(device)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.float).to(device).unsqueeze(1)

    layers = [torch.nn.Linear(2, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
    model = torch.nn.Sequential(*layers).to(device)
    validate_network(model)

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
            model = prune_equal_fanin(model, 2, device=device)

    # generate explanations
    weights, biases = collect_parameters(model, device=device)
    f = fol.generate_fol_explanations(weights, biases)[0]
    print(f'Explanation: {f}')

    return


if __name__ == "__main__":
    main()
