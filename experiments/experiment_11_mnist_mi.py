
if __name__ == "__main__":
    #%%

    import sys
    import time

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    import deep_logic as dl
    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
    from deep_logic.utils.metrics import ClusterAccuracy, F1Score
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.utils.datasets import ConceptOnlyDataset
    from deep_logic.utils.data import get_splits_train_val_test
    from deep_logic.utils.loss import MutualInformationLoss
    from deep_logic.logic import test_explanation, fidelity, complexity
    from data import MNIST
    from data.download_mnist import download_mnist
    from experiments.MNIST.concept_extractor_mnist import concept_extractor_mnist

    results_dir = 'results/mnist_mi'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #%% md
    ## Loading MNIST data
    #%%

    dataset_root = "../data/MNIST_EVEN_ODD/"
    if not os.path.isdir(dataset_root):
        download_mnist(dataset_root)
    else:
        print("Dataset already downloaded")
    print(dataset_root)

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
    l_r = 1e-3
    lr_scheduler = True
    simplify = True
    seeds = [*range(10)]
    print("Seeds", seeds)
    top_k_explanations = 5
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    #%% md
    ## Define methods, loss, metrics and saved metrics
    #%%

    method_list = ['Psi', 'General', 'Relu']
    loss = MutualInformationLoss()
    metric = ClusterAccuracy()

    for method in method_list:

        methods = []
        splits = []
        explanations = []
        model_accuracies = []
        explanation_accuracies = []
        elapsed_times = []
        explanation_fidelities = []
        explanation_complexities = []

        for seed in seeds:
            set_seed(seed)
            name = os.path.join(results_dir, f"{method}_{seed}")

            train_data, val_data, test_data = get_splits_train_val_test(dataset, load=False)
            x_val = torch.tensor(dataset.attributes[val_data.indices])
            y_val = torch.tensor(dataset.targets[val_data.indices])
            x_test = torch.tensor(dataset.attributes[test_data.indices])
            y_test = torch.tensor(dataset.targets[test_data.indices])
            print(train_data.indices)

            # Setting device
            print(f"Training {name} Classifier...")
            start_time = time.time()

            if method == 'Psi':
                # Network structures
                l1_weight = 1e-2
                print("l1 weight", l1_weight)
                hidden_neurons = []
                fan_in = 2
                lr_psi = 1e-2
                model = PsiNetwork(n_clusters, n_features, hidden_neurons, loss,
                                   l1_weight, name=name, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_predictions, exp_complexities = [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(i, concept_names, simplify=simplify)
                    _, exp_prediction = test_explanation(formula, i, x_test, y_test,
                                                         metric=F1Score(), concept_names=concept_names)
                    exp_prediction = torch.as_tensor(exp_prediction)
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula)
                    exp_predictions.append(exp_prediction)
                    exp_complexities.append(explanation_complexity)
                    print(f"Formula {i}: {formula}")
                    print("Explanation complexity", explanation_complexity)
                outputs = outputs.argmax(dim=1)
                exp_predictions = torch.stack(exp_predictions, dim=1)
                exp_accuracy = metric(exp_predictions, labels)
                exp_fidelity = fidelity(exp_predictions, outputs, metric)

            elif method == 'General':
                # Network structures
                l1_weight = 1e-3
                fan_in = None
                hidden_neurons = [20, 10]
                model = XGeneralNN(n_classes=n_clusters, n_features=n_features, hidden_neurons=hidden_neurons,
                                   loss=loss, name=name, l1_weight=l1_weight, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                                        lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_predictions, exp_complexities = [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_val, y_val, i, simplify=simplify,
                                                           topk_explanations=top_k_explanations,
                                                           concept_names=concept_names)
                    _, exp_prediction = test_explanation(formula, i, x_test, y_test,
                                                         metric=F1Score(), concept_names=concept_names)
                    exp_prediction = torch.as_tensor(exp_prediction)
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula)
                    exp_predictions.append(exp_prediction)
                    exp_complexities.append(explanation_complexity)
                    print(f"Formula {i}: {formula}")
                    print("Explanation complexity", explanation_complexity)
                outputs = outputs.argmax(dim=1)
                exp_predictions = torch.stack(exp_predictions, dim=1)
                exp_accuracy = metric(exp_predictions, labels)
                exp_fidelity = fidelity(exp_predictions, outputs, metric)

            elif method == 'Relu':
                # Network structures
                l1_weight = 1e-4
                hidden_neurons = [50, 30]
                model = XReluNN(n_classes=n_clusters, n_features=n_features, name=name,
                                hidden_neurons=hidden_neurons, loss=loss, l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_predictions, exp_complexities = [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_val, y_val, i, simplify=simplify,
                                                           topk_explanations=top_k_explanations,
                                                           concept_names=concept_names)
                    _, exp_prediction = test_explanation(formula, i, x_test, y_test,
                                                         metric=F1Score(), concept_names=concept_names)
                    exp_prediction = torch.as_tensor(exp_prediction)
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula)
                    exp_predictions.append(exp_prediction)
                    exp_complexities.append(explanation_complexity)
                    print(f"Formula {i}: {formula}")
                    print("Explanation complexity", explanation_complexity)
                outputs = outputs.argmax(dim=1)
                exp_predictions = torch.stack(exp_predictions, dim=1)
                exp_accuracy = accuracy_score(exp_predictions, labels, metric)
                exp_fidelity = fidelity(exp_predictions, outputs, metric)

            else:
                raise NotImplementedError(f"{method} not implemented")

            elapsed_time = time.time() - start_time
            methods.append(method)
            splits.append(seed)
            explanations.append(formulas[0])
            model_accuracies.append(accuracy)
            explanation_accuracies.append(exp_accuracy)
            explanation_fidelities.append(exp_fidelity)
            elapsed_times.append(elapsed_time)
            explanation_complexities.append(np.mean(exp_complexities))

        explanation_consistency = dl.logic.formula_consistency(explanations)
        print(f'Consistency of explanations: {explanation_consistency:.4f}')

        results = pd.DataFrame({
            'method': methods,
            'split': splits,
            'explanation': explanations,
            'model_accuracy': model_accuracies,
            'explanation_accuracy': explanation_accuracies,
            'explanation_fidelity': explanation_fidelities,
            'explanation_complexity': explanation_complexities,
            'explanation_consistency': explanation_consistency,
            'elapsed_time': elapsed_times,
        })
        results.to_csv(os.path.join(results_dir, f'results_{method}.csv'))
        print(results)

    #%% md
    ##Summary
    #%%

    cols = ['model_accuracy', 'explanation_accuracy', 'explanation_fidelity', 'explanation_complexity', 'elapsed_time',
            'explanation_consistency']
    mean_cols = [f'{c}_mean' for c in cols]
    sem_cols = [f'{c}_sem' for c in cols]

    results = {}
    summaries = {}
    for method in method_list:
        results[method] = pd.read_csv(os.path.join(results_dir, f"results_{method}.csv"))
        df_mean = results[method][cols].mean()
        df_sem = results[method][cols].sem()
        df_mean.columns = mean_cols
        df_sem.columns = sem_cols
        summaries[method] = pd.concat([df_mean, df_sem])
        summaries[method].name = method

    results = pd.concat([results[method] for method in method_list], axis=1).T
    results.to_csv(os.path.join(results_dir, f'results.csv'))

    summary = pd.concat([summaries[method] for method in method_list], axis=1).T
    summary.columns = mean_cols + sem_cols
    summary.to_csv(os.path.join(results_dir, 'summary.csv'))
    print(summary)
