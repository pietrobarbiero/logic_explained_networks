
if __name__ == "__main__":
    #%%

    import sys

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.utils.base import set_seed
    from deep_logic.utils.metrics import UnsupervisedMetric
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.utils.datasets import ConceptOnlyDataset
    from deep_logic.utils.data import get_splits_train_val_test
    from deep_logic.utils.loss import MutualInformationLoss
    from data import MNIST
    from data.download_mnist import download_mnist
    from experiments.MNIST.concept_extractor_mnist import concept_extractor_mnist

    results_dir = 'results/mnist'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #%% md
    ## Loading MNIST data
    #%%

    dataset_root = f"../data/MNIST/"
    if not os.path.isdir(dataset_root):
        download_mnist(dataset_root)
    else:
        print("Dataset already downloaded")

    #%% md
    ## Extracting concepts
    #%%

    if not os.path.isfile(os.path.join(dataset_root, f"{MNIST}_multi_label_predictions.npy")):
        concept_extractor_mnist(dataset_root, multi_label=True)
    else:
        print("Concepts already extracted")
    dataset = ConceptOnlyDataset(dataset_root, dataset_name=MNIST)
    concept_names = dataset.attribute_names
    print("Concept names", concept_names)
    n_features = dataset.n_attributes
    print("Number of features", n_features)
    n_clusters = 2
    print("Number of cluster", n_clusters)

    #%% md
    ## Training Hyperparameters
    #%%

    epochs = 100
    l_r = 0.001
    lr_scheduler = True
    simplify = True
    seeds = [*range(10)]
    print("Seeds", seeds)
    top_k_explanations = 5

    #%% md
    ## Define loss, metrics and saved metrics
    #%%

    loss = MutualInformationLoss()
    metric = UnsupervisedMetric()

    methods = []
    splits = []
    explanations = []
    explanations_inv = []
    elapsed_times = []
    elapsed_times_inv = []

    #%% md
    ## Relu NN
    #%%

    for seed in seeds:
        print("Seed", seed)
        set_seed(seed)

        train_data, val_data, test_data = get_splits_train_val_test(dataset)
        print(train_data.indices)

        x_val = torch.tensor(dataset.attributes[val_data.indices])

        # Network structures
        l1_weight = 1e-5
        hidden_neurons = [100, 10]

        # Setting device
        device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)
        print(f"Training Relu NN...")
        model = XReluNN(n_classes=dataset.n_classes, n_features=n_features,
                        hidden_neurons=hidden_neurons, loss=loss, l1_weight=l1_weight)

        results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                            metric=metric, lr_scheduler=lr_scheduler, device=device, save=False)
        preds, _ = model.predict(val_data)
        print(results)

        formulas, times = [], []
        for i, class_to_explain in enumerate(dataset.classes):
            formula, elapsed_time = model.get_global_explanation(x_val, preds, i,
                                                                 topk_explanations=top_k_explanations,
                                                                 concept_names=concept_names,
                                                                 simplify=True, return_time=True)
            formulas.append(formula), times.append(elapsed_time)
            print(f"{formula}")
            print("Elapsed time", elapsed_time)

        methods.append("Relu")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    results = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'elapsed_time': elapsed_times,
        'elapsed_time_inv': elapsed_times_inv,
    })
    results_relu = results[results['method'] == "Relu"]
    results_relu.to_csv(os.path.join(results_dir, 'results_relu.csv'))
    print(results_relu)

    #%% md
    ## PSI NN
    #%%

    for seed in seeds:
        print("Seed", seed)
        set_seed(seed)

        train_data, val_data, test_data = get_splits_train_val_test(dataset)
        print(train_data.indices)

        # Network structures
        l1_weight = 5e-6
        hidden_neurons = []
        fan_in = 6
        lr_psi = 0.001
        n_cluster = 2

        # Setting device
        device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)

        print("Training Psi NN...")
        model = PsiNetwork(n_cluster, n_features, hidden_neurons, loss,
                           l1_weight, fan_in=fan_in)

        results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                            metric=metric, lr_scheduler=lr_scheduler, device=device, save=False)
        print(results)

        formulas, times = [], []
        for i in range(n_cluster):
            formula, elapsed_time = model.get_global_explanation(i, concept_names,
                                                                 simplify=True, return_time=True)
            formulas.append(formula), times.append(elapsed_time)
            print(f"{i}) - {formula}")
            print("Elapsed time", elapsed_time)

        methods.append("Psi")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    results = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'elapsed_time': elapsed_times,
        'elapsed_time_inv': elapsed_times_inv,
    })
    results_psi = results[results['method'] == "Psi"]
    results_psi.to_csv(os.path.join(results_dir, 'results_psi.csv'))
    print(results_psi)

    #%% md
    ## Mu NN
    #%%

    for seed in seeds:
        print("Seed", seed)
        set_seed(seed)

        train_data, val_data, test_data = get_splits_train_val_test(dataset)
        print(train_data.indices)

        x_val = torch.tensor(dataset.attributes[val_data.indices])

        # Network structures
        l1_weight = 1e-3
        hidden_neurons = [20, 10]
        fan_in = 6

        # Setting device
        device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)

        print("Training General NN...")
        model = XGeneralNN(n_classes=dataset.n_classes, n_features=n_features, hidden_neurons=hidden_neurons,
                           loss=loss, l1_weight=l1_weight, fan_in=fan_in)

        results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                            lr_scheduler=lr_scheduler, device=device, save=False, verbose=True)
        print(results)

        preds, _ = model.predict(val_data)
        formulas, times = [], []
        for i, class_to_explain in enumerate(dataset.classes):
            formula, elapsed_time = model.get_global_explanation(x_val, preds, i, simplify=True,
                                                                 topk_explanations=top_k_explanations,
                                                                 concept_names=concept_names, return_time=True)
            formulas.append(formula), times.append(elapsed_time)
            print(f"{formula}")
            print("Elapsed time", elapsed_time)

        methods.append("General")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    #%%

    results = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'elapsed_time': elapsed_times,
        'elapsed_time_inv': elapsed_times_inv,
    })
    results_general = results[results['method'] == "General"]
    results_general.to_csv(os.path.join(results_dir, 'results_general.csv'))
    results.to_csv(os.path.join(results_dir, 'results.csv'))
    print(results)

