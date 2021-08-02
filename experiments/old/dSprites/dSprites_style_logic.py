import torch
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import os
import copy
import lens as dl


def logic_learning(net_top, dataloader, device, optimizer, criterion, epoch, n_epochs, train=False):
    running_loss = 0.0
    y_correct = 0
    y_total = 0
    total_step = len(dataloader)
    for batch_idx, (concepts_, target_) in enumerate(dataloader):
        concepts_, target_ = concepts_.to(device), target_.to(device)

        if train:
            optimizer.zero_grad()

        y_preds = net_top(concepts_)
        loss = criterion(y_preds, target_)
        # loss = 0
        # for task_id in range(y_preds.size(1)):
        #     y_preds_i = y_preds[:, task_id]
        #     target_i = target_[:, task_id]
        #     min_samples = torch.min(sum(target_i), len(target_i) - sum(target_i))
        #     min_samples = int(min_samples.item())
        #     a = torch.where(target_i==0)[0]
        #     b = torch.where(target_i==1)[0]
        #     c = torch.randperm(min_samples)[:min_samples]
        #     ac = a[c.detach().numpy()]
        #     bc = b[c.detach().numpy()]
        #     abc = torch.hstack([ac, bc])
        #     y_preds_i_abc = y_preds_i[abc]
        #     target_i_abc = target_i[abc]
        #
        #     loss += criterion(y_preds_i_abc, target_i_abc)

        for module in net_top.children():
            if isinstance(module, torch.nn.Linear):
                loss += 0.001 * torch.norm(module.weight, 1)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        y_correct += torch.sum((y_preds > 0.5).eq(target_).sum(dim=1) == target_.size(1)).item()
        y_total += target_.size(0)
        # if (batch_idx) % 20 == 0 and train:
        #     print(f'Epoch [{epoch}/{n_epochs}], Step [{batch_idx}/{total_step}], Loss: {loss.item():.4f}')

    y_accuracy = y_correct / y_total
    loss = running_loss / total_step

    # for module in net_top.children():
    #     if isinstance(module, torch.nn.Linear):
    #         loss += 0.01 * torch.norm(module.weight, 1)

    return net_top, y_accuracy, loss


def logic_style(train_dataloader, val_dataloader, test_dataloader, device, args):
    concepts_, target_ = next(iter(train_dataloader))
    n_classes_c, n_classes_y = concepts_.size(1), target_.size(1)
    h_size = 10

    net_top = torch.nn.Sequential(*[
        torch.nn.Linear(n_classes_c, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, n_classes_y),
        torch.nn.Sigmoid(),
    ]).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net_top.parameters(), lr=0.01)

    n_epochs = 1000
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []
    print('Start training')
    for epoch in range(1, n_epochs + 1):
        net_top, y_accuracy, loss = logic_learning(net_top, train_dataloader,
                                                   device, optimizer, criterion,
                                                   epoch, n_epochs, train=True)
        train_acc.append(y_accuracy)
        train_loss.append(loss)
        train_loss_to_save = loss

        with torch.no_grad():
            net_top.eval()

            net_top, y_accuracy, loss = logic_learning(net_top, val_dataloader,
                                                       device, optimizer, criterion,
                                                       epoch, n_epochs, train=False)
            val_acc.append(y_accuracy)
            val_loss.append(loss)

            if loss < valid_loss_min and epoch % 100 == 0:
                valid_loss_min = loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net_top.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_to_save,
                }, os.path.join(args.models_dir, f'mlp_{args.model_style}_c2y_{args.seed}.pt'))

            net_top, y_accuracy, loss = logic_learning(net_top, test_dataloader,
                                                       device, optimizer, criterion,
                                                       epoch, n_epochs, train=False)
            test_acc.append(y_accuracy)
            test_loss.append(loss)

        net_top.train()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}\n\ttrain-loss: {loss:.4f}, train-y-acc: {y_accuracy:.4f}')
            print(f'\tvalidation loss: {loss:.4f}, validation-y-acc: {y_accuracy:.4f}')
            print(f'\ttest loss: {loss:.4f}, test-y-acc: {y_accuracy:.4f}\n')

    results = pd.DataFrame({
        'test_acc': test_acc,
        'test_loss': test_loss,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
    })

    return net_top, results
