import torch
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from .base import collect_parameters, to_categorical
from .relunn import get_reduced_model


def rank_pruning(model, x_sample, y, device):
    # get the model prediction on the individual sample
    y_pred_sample = model(x_sample)
    pred_class = to_categorical(y_pred_sample)
    y = to_categorical(y)
    n_classes = len(torch.unique(y))

    # identify non-pruned features
    w, b = collect_parameters(model, device)
    feature_weights = w[0]
    block_size = feature_weights.shape[0] // n_classes
    feature_used_bool = np.sum(np.abs(feature_weights[pred_class * block_size:(pred_class + 1) * block_size]),
                               axis=0) > 0
    feature_used = np.sort(np.nonzero(feature_used_bool)[0])
    return feature_used


def rank_weights(model, x_sample, device):
    reduced_model = get_reduced_model(model, x_sample.squeeze())
    w, b = collect_parameters(reduced_model, device)
    w_abs = torch.norm(torch.FloatTensor(w[0]), dim=0)
    w_max = torch.max(w_abs)
    w_bool = ((w_abs / w_max) > 0.5).cpu().detach().numpy().squeeze()
    if sum(w_bool) == 0:
        return ''

    feature_used = np.sort(np.nonzero(w_bool)[0])
    return feature_used


def rank_lime(model, x_train, x_sample, num_features, device):
    def _torch_predict(x):
        model.eval()
        preds = model(torch.FloatTensor(x).to(device)).cpu().detach().numpy()
        if len(preds.squeeze().shape) == 1:
            predictions = np.zeros((preds.shape[0], 2))
            predictions[:, 0] = 1 - preds.squeeze()
            predictions[:, 1] = preds.squeeze()
        else:
            predictions = preds
        return predictions

    explainer = LimeTabularExplainer(x_train.cpu().detach().numpy(), discretize_continuous=True)
    exp = explainer.explain_instance(x_sample.cpu().detach().numpy().squeeze(), _torch_predict,
                                     num_features=num_features, top_labels=1)

    feature_used = []
    weights = []
    feature_ids = []
    for _, v in exp.local_exp.items():
        for feature_id, w in v:
            feature_ids.append(feature_id)
            weights.append(w)

    feature_weights = np.array(weights)
    normalized_weights = np.abs(feature_weights) / np.max(np.abs(feature_weights))

    for feature_id, normalized_weight in zip(feature_ids, normalized_weights):
        if normalized_weight > 0.1:
            feature_used.append(feature_id)

    return np.sort(np.array(feature_used))
