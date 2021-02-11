import torch
from torchvision.models import resnet18
import numpy as np
import pandas as pd
import os


def logic_learning(i2c_net, dataloader, device, optimizer, criterion, epoch, n_epochs, train=False):
    running_loss = 0.0
    c_correct = 0
    c_total = 0
    preds = []
    total_step = len(dataloader)
    for batch_idx, (data_, concepts_) in enumerate(dataloader):
        data_, concepts_ = data_.to(device), concepts_.to(device)

        if train:
            optimizer.zero_grad()

        c_preds = i2c_net(data_)
        loss = criterion(c_preds, concepts_)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        c_correct += torch.sum((c_preds > 0.5).eq(concepts_).sum(dim=1) == concepts_.size(1)).item()
        c_total += concepts_.size(0)
        preds.append(c_preds.cpu().detach().numpy())

        if (batch_idx) % 20 == 0 and train:
            print(f'Epoch [{epoch}/{n_epochs}], Step [{batch_idx}/{total_step}], Loss: {loss.item():.4f}')

    c_accuracy = c_correct / c_total
    loss = running_loss / total_step

    return i2c_net, c_accuracy, loss, preds


def load_and_predict(i2c_net, train_dataloader, val_dataloader, test_dataloader,
                     optimizer, criterion, device, args):
    i2c_net.load_state_dict(torch.load(os.path.join(args['models_dir'], f'resnet_{args["model_style"]}_i2c_{args["seed"]}.pt')))
    i2c_net.eval()

    _, _, _, c_predictions_train = logic_learning(i2c_net, train_dataloader,
                                                  device, optimizer, criterion,
                                                  1, 1, train=False)
    c_predictions_train = (np.vstack(c_predictions_train)>0.5).astype(int)

    _, _, _, c_predictions_val = logic_learning(i2c_net, val_dataloader,
                                                device, optimizer, criterion,
                                                1, 1, train=False)
    c_predictions_val = (np.vstack(c_predictions_val)>0.5).astype(int)

    _, _, _, c_predictions_test = logic_learning(i2c_net, test_dataloader,
                                                 device, optimizer, criterion,
                                                 1, 1, train=False)
    c_predictions_test = (np.vstack(c_predictions_test)>0.5).astype(int)
    return c_predictions_train, c_predictions_val, c_predictions_test


def i2c_style(train_dataloader, val_dataloader, test_dataloader, device, args):
    data_, concepts_ = next(iter(train_dataloader))
    n_classes_c = concepts_.size(1)

    i2c_net = resnet18(pretrained=False)
    num_ftrs = i2c_net.fc.in_features
    i2c_net.fc = torch.nn.Linear(num_ftrs, n_classes_c)
    i2c_net = torch.nn.Sequential(*[
        i2c_net,
        torch.nn.Sigmoid(),
    ])
    i2c_net.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(i2c_net.parameters(), lr=0.0001)

    if os.path.isfile(os.path.join(args['models_dir'], f'resnet_{args["model_style"]}_i2c_{args["seed"]}.pt')):
        i2c_net.load_state_dict(
            torch.load(os.path.join(args['models_dir'], f'resnet_{args["model_style"]}_i2c_{args["seed"]}.pt')))
        c_predictions_train, c_predictions_val, c_predictions_test = load_and_predict(i2c_net, train_dataloader,
                                                                                      val_dataloader, test_dataloader,
                                                                                      optimizer, criterion, device,
                                                                                      args)
        return c_predictions_train, c_predictions_val, c_predictions_test

    n_epochs = 200
    valid_loss_min = np.Inf
    val_loss = []
    val_c_acc = []
    test_loss = []
    test_c_acc = []
    train_loss = []
    train_c_acc = []
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}\n')

        i2c_net, c_accuracy, loss, _ = logic_learning(i2c_net, train_dataloader,
                                                      device, optimizer, criterion,
                                                      epoch, n_epochs, train=True)
        train_c_acc.append(c_accuracy)
        train_loss.append(loss)
        print(f'\ntrain-loss: {loss:.4f}, train-c-acc: {c_accuracy:.4f}')

        with torch.no_grad():
            i2c_net.eval()

            i2c_net, c_accuracy, loss, _ = logic_learning(i2c_net, val_dataloader,
                                                          device, optimizer, criterion,
                                                          epoch, n_epochs, train=False)
            val_c_acc.append(c_accuracy)
            val_loss.append(loss)
            print(f'validation loss: {loss:.4f}, validation-c-acc: {c_accuracy:.4f}')

            if loss < valid_loss_min:
                valid_loss_min = loss
                torch.save(i2c_net.state_dict(),
                           os.path.join(args['models_dir'], f'resnet_{args["model_style"]}_i2c_{args["seed"]}.pt'))
                print('Improvement-Detected, save-model')

            i2c_net, c_accuracy, loss, _ = logic_learning(i2c_net, test_dataloader,
                                                          device, optimizer, criterion,
                                                          epoch, n_epochs, train=False)
            test_c_acc.append(c_accuracy)
            test_loss.append(loss)
            print(f'test loss: {loss:.4f}, test-c-acc: {c_accuracy:.4f}\n')

        i2c_net.train()

    results = pd.DataFrame({
        'test_c_acc': test_c_acc,
        'test_loss': test_loss,
        'val_c_acc': val_c_acc,
        'val_loss': val_loss,
        'train_c_acc': train_c_acc,
        'train_loss': train_loss,
    })

    results_dir = './results'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results.to_csv(os.path.join(results_dir, f'results_{args["model_style"]}_i2c_{args["seed"]}.csv'))

    # i2c_net.load_state_dict(torch.load(os.path.join(args['models_dir'], f'resnet_{args["model_style"]}_i2c_{args["seed"]}.pt')))
    c_predictions_train, c_predictions_val, c_predictions_test = load_and_predict(i2c_net, train_dataloader,
                                                                                  val_dataloader, test_dataloader,
                                                                                  optimizer, criterion, device, args)

    return c_predictions_train, c_predictions_val, c_predictions_test
