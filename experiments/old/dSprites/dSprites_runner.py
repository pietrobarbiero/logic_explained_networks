import sys
sys.path.append('../..')

import os
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import lens as dl

from dSprites_style_CBM import cbm_style
from dSprites_style_E2E import e2e_style
from dSprites_loader import get_data
from dSprites_style_logic import logic_style
from dSprites_style_I2C import i2c_style

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main(args):

    if not os.path.isdir(args.models_dir):
        os.makedirs(args.models_dir)

    if os.path.isfile(args.models_dir):
        net = torch.load(os.path.join(args.models_dir, f'resnet_{args.model_style}_{args.seed}.pt'))
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # Load the model, as well as input, label, and concept data
    x_train, y_train, x_val, y_val, x_test, y_test, c_train, c_val, c_test, c_names = get_data(args)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_val = torch.FloatTensor(x_val)
    y_val = torch.FloatTensor(y_val)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    c_train = torch.FloatTensor(c_train)
    c_val = torch.FloatTensor(c_val)
    c_test = torch.FloatTensor(c_test)
    print("Data loaded successfully...")

    batch_size = 128

    if args.model_style == 'E2E':
        train_dataset = TensorDataset(x_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(x_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        n_classes = len(torch.unique(y_train))
        net, results = e2e_style(train_dataloader, val_dataloader, test_dataloader, n_classes, device, args)

    elif args.model_style == 'CBM':
        train_dataset = TensorDataset(x_train, c_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(x_val, c_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(x_test, c_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        net, results = cbm_style(train_dataloader, val_dataloader, test_dataloader, device, args)

    elif args.model_style == 'logic':
        train_dataset = TensorDataset(x_train, c_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(x_val, c_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(x_test, c_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        c_predictions_train, c_predictions_val, c_predictions_test = i2c_style(train_dataloader, val_dataloader, test_dataloader, device, args)
        c_predictions_train = torch.FloatTensor(c_predictions_train)
        c_predictions_val = torch.FloatTensor(c_predictions_val)
        c_predictions_test = torch.FloatTensor(c_predictions_test)

        train_batch_size = len(x_train)
        val_batch_size = len(x_val)
        test_batch_size = len(x_test)

        print(f'Accuracy train: {accuracy_score(c_predictions_train, c_train)}')
        print(f'Accuracy val: {accuracy_score(c_predictions_val, c_val)}')
        print(f'Accuracy test: {accuracy_score(c_predictions_test, c_test)}')

        y_train = y_train[:, 0].unsqueeze(1)
        y_val = y_val[:, 0].unsqueeze(1)
        y_test = y_test[:, 0].unsqueeze(1)

        train_dataset = TensorDataset(c_predictions_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
        val_dataset = TensorDataset(c_predictions_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_dataset = TensorDataset(c_predictions_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        model, results = logic_style(train_dataloader, val_dataloader, test_dataloader, device, args)

        results_dir = './results'
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        results.to_csv(os.path.join(results_dir, f'results_{args.model_style}_c2y_{args.seed}.csv'))

        # generate explanations
        local_explanations = []
        for i, (xin, yin) in enumerate(zip(c_test, y_test)):
            model_reduced = lens.get_reduced_model(model, xin.to(device))
            output = model_reduced(xin)
            if output > 0.5 and (output > 0.5) == yin:
                local_explanation = lens.fol.generate_local_explanations(model_reduced, xin, k=2)
                local_explanations.append(local_explanation)

        print(local_explanations)
        global_explanation, predictions = lens.fol.combine_local_explanations(model.cpu(), c_test.cpu(), y_test.cpu(), k=2)
        print(global_explanation)

        ynp = y_test.detach().numpy()[:, 0]
        accuracy = np.sum(predictions == ynp) / len(ynp)
        print(accuracy)

        explanations = {
            "local explanations": local_explanations,
            "global explanations": global_explanation,
            "test predictions": predictions,
            "explanation accuracy": accuracy,
        }
        np.save(os.path.join(results_dir, f'explanations_{args.model_style}_c2y_{args.seed}'), explanations)

        # read_dictionary = np.load(os.path.join(results_dir, f'explanations_{args.model_style}_c2y_{args.seed}.npy'), allow_pickle='TRUE').item()

    return


if __name__ == '__main__':

    seed = 0
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()

    parser.add_argument('-a0', '--seed', type=int, default=seed, help='Random seed')
    parser.add_argument('-a1', '--task_name', type=str, choices=['shape', 'shape_scale'], default='shape_scale', help='Name of dSprites task to run')
    parser.add_argument('-a2', '--itc_method', type=str, choices=['cme', 'net2vec'], default='cme', help='Type of concept extractor to use. '
                                                                                                  'Choices are to either use CME, or Net2Vec.')
    parser.add_argument('-a3', '--itc_model', type=str, choices=['LR', 'LP'], default='LP',
                                        help='Type of model to use for predicting concept values. Is either Linear Regression (LR), '
                                             'or Label Propagation (LP).')
    parser.add_argument('-a4', '--ctl_model', type=str, choices=['DT', 'LR'], default='DT',
                                        help='Type of model to use for predicting task labels. Is either Linear Regression (LR), '
                                             'or Decision Tree (DT).')
    parser.add_argument('-a5', '--start_layer', type=int, default=0, help='Layer idx of first layer from which to perform concept extraction')
    parser.add_argument('-a6', '--batch_size_extract', type=int, default=128, help='Batch size to use during concept extraction')
    parser.add_argument('-a7', '--n_labelled', type=int, default=100, help='Number of labelled samples to use for experiments')
    parser.add_argument('-a8', '--n_unlabelled', type=int, default=200, help='Number of unlabelled samples to use for experiments')
    parser.add_argument('-a9', '--tsne_viz', type=bool, default=False, help='Whether to plot the tSNE figure')
    parser.add_argument('-a10', '--n_tsne_samples', type=int, default=1000, help='Number of samples to use for tSNE plot')
    parser.add_argument('-a11', '--figs_path', type=str, default=None, help='Directory path for where to save the figures')
    parser.add_argument('-a12', '--dsprites_path', type=str, default='./data/dsprites.npz', help='Path to the dSprites data file')
    parser.add_argument('-a13', '--models_dir', type=str, default='./models', help='Path where models are saved/loaded from')
    parser.add_argument('-a14', '--model_style', type=str, default='logic',
                        choices=['E2E', 'CBM', 'logic'], help='Task')

    args = parser.parse_args()

    print(main(args))



