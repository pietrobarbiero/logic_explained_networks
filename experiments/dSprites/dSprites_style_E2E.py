import torch
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import os


def e2e_style(train_dataloader, val_dataloader, test_dataloader, n_classes, device, args):
    net = resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, n_classes)
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    n_epochs = 25
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}\n')
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')

        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (val_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(val_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), os.path.join(args.models_dir, f'resnet_{args.model_style}_{args.seed}.pt'))
                print('Improvement-Detected, save-model')

            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            test_acc.append(100 * correct_t / total_t)
            test_loss.append(batch_loss / len(test_dataloader))
            print(f'test loss: {np.mean(test_loss):.4f}, test acc: {(100 * correct_t / total_t):.4f}\n')

        net.train()

    results = pd.DataFrame({
        'test_loss': test_loss,
        'test_acc': test_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'train_acc': train_acc,
    })

    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results.to_csv(os.path.join(results_dir, f'results_{args.model_style}_{args.seed}.csv'))

    return net, results
