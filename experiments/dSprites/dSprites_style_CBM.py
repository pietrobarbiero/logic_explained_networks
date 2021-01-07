import torch
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import os


def cbm_learning(net, net_base, dataloader, device, optimizer, criterion, epoch, n_epochs, train=False):
    running_loss = 0.0
    y_correct = 0
    c_correct = 0
    c_total = 0
    y_total = 0
    total_step = len(dataloader)
    for batch_idx, (data_, concepts_, target_) in enumerate(dataloader):
        data_, concepts_, target_ = data_.to(device), concepts_.to(device), target_.to(device)

        if train:
            optimizer.zero_grad()

        c_preds = net_base(data_)
        y_preds = net(data_)
        loss = criterion(y_preds, target_) + 1 * criterion(c_preds, concepts_)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        c_correct += torch.sum((c_preds > 0.5).eq(concepts_)).item()
        y_correct += torch.sum((y_preds > 0.5).eq(target_)).item()
        c_total += concepts_.size(0) * concepts_.size(1)
        y_total += target_.size(0) * target_.size(1)
        if (batch_idx) % 20 == 0 and train:
            print(f'Epoch [{epoch}/{n_epochs}], Step [{batch_idx}/{total_step}], Loss: {loss.item():.4f}')

    c_accuracy = c_correct / c_total
    y_accuracy = y_correct / y_total
    loss = running_loss / total_step

    return net, net_base, c_accuracy, y_accuracy, loss

def cbm_style(train_dataloader, val_dataloader, test_dataloader, device, args):
    data_, concepts_, target_ = next(iter(train_dataloader))
    n_classes_c, n_classes_y = concepts_.size(1), target_.size(1)
    net_base = resnet18(pretrained=False)
    num_ftrs = net_base.fc.in_features
    net_base.fc = torch.nn.Linear(num_ftrs, n_classes_c)
    net_base  = torch.nn.Sequential(*[
        net_base,
        torch.nn.Sigmoid(),
    ])
    n_top = torch.nn.Sequential(*[
        torch.nn.Linear(n_classes_c, n_classes_y),
        torch.nn.Sigmoid()
    ])
    net = torch.nn.Sequential(*[net_base, n_top])
    net = net.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    n_epochs = 200
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    val_c_acc = []
    test_loss = []
    test_acc = []
    test_c_acc = []
    train_loss = []
    train_acc = []
    train_c_acc = []
    epoch = 1
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}\n')

        net, net_base, c_accuracy, y_accuracy, loss = cbm_learning(net, net_base, train_dataloader,
                                                                   device, optimizer, criterion,
                                                                   epoch, n_epochs, train=True)
        train_c_acc.append(c_accuracy)
        train_acc.append(y_accuracy)
        train_loss.append(loss)
        print(f'\ntrain-loss: {loss:.4f}, train-c-acc: {c_accuracy:.4f}, train-cy-acc: {y_accuracy:.4f}')

        with torch.no_grad():
            net.eval()

            net, net_base, c_accuracy, y_accuracy, loss = cbm_learning(net, net_base, val_dataloader,
                                                                       device, optimizer, criterion,
                                                                       epoch, n_epochs, train=False)
            val_c_acc.append(c_accuracy)
            val_acc.append(y_accuracy)
            val_loss.append(loss)
            print(f'validation loss: {loss:.4f}, validation-c-acc: {c_accuracy:.4f}, validation-y-acc: {y_accuracy:.4f}')

            if loss < valid_loss_min:
                valid_loss_min = loss
                torch.save(net.state_dict(), os.path.join(args.models_dir, f'resnet_{args.model_style}_{args.seed}.pt'))
                print('Improvement-Detected, save-model')

            net, net_base, c_accuracy, y_accuracy, loss = cbm_learning(net, net_base, test_dataloader,
                                                                       device, optimizer, criterion,
                                                                       epoch, n_epochs, train=False)
            test_c_acc.append(c_accuracy)
            test_acc.append(y_accuracy)
            test_loss.append(loss)
            print(f'test loss: {loss:.4f}, test-c-acc: {c_accuracy:.4f}, test-y-acc: {y_accuracy:.4f}\n')

        net.train()

    results = pd.DataFrame({
        'test_acc': test_acc,
        'test_c_acc': test_c_acc,
        'test_loss': test_loss,
        'val_acc': val_acc,
        'val_c_acc': val_c_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_c_acc': train_c_acc,
        'train_loss': train_loss,
    })

    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results.to_csv(os.path.join(results_dir, f'results_{args.model_style}_{args.seed}.csv'))

    return net, results
