from utils.data import clean_names

if __name__ == "__main__":
    #%%

    import sys
    import time
    import ast

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import deep_logic as dl
    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
    from deep_logic.utils.metrics import F1Score, ClusterAccuracy
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.logic import test_explanation, complexity, fidelity
    from deep_logic.logic.metrics import accuracy_score
    from deep_logic.utils.loss import MutualInformationLoss
    from torch.utils.data import TensorDataset

    results_dir = 'results/celldiff_mi'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #%% md
    ## Loading CellDiff data
    #%%

    dataset_root = "../data/celldiff/"
    print(dataset_root)

    gene_expression_matrix = pd.read_csv(os.path.join(dataset_root,
                                                      'data_matrix.csv'), index_col=0)
    clustering_labels = pd.read_csv(os.path.join(dataset_root,
                                                 'cluster_labels.csv'), index_col=0)
    biomarkers = pd.read_csv(os.path.join(dataset_root,
                                          'markers.csv'), index_col=0)

    markers = []
    for _, row in biomarkers.iterrows():
        print(row)
        markers.extend(ast.literal_eval(row['markers']))
    # get unique markers
    markers = set(markers)

    x_np = gene_expression_matrix[markers].values
    y_np = clustering_labels.values

    scaler = MinMaxScaler((0, 1))
    x_np_scaled = scaler.fit_transform(x_np)

    x = torch.FloatTensor(x_np_scaled)
    y = torch.FloatTensor(y_np)

    dataset = TensorDataset(x, y)
    dataset.attribute_names = markers
    dataset.n_attributes = len(markers)
    dataset.classes = np.unique(y_np)

    concept_names = dataset.attribute_names
    concept_names = clean_names(concept_names)
    print("Concept names", concept_names)
    n_features = dataset.n_attributes
    print("Number of features", n_features)
    n_clusters = len(np.unique(y_np))
    print("Number of cluster", n_clusters)

    #%% md
    ## Training Hyperparameters
    #%%

    epochs = 10000
    l_r = 1e-4
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

    method_list = ['Relu', 'Psi', 'General']
    loss = MutualInformationLoss(penalize_inactive=True)
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

            dataset_size = len(dataset)
            train_len = int(0.8*dataset_size)
            val_len = int(0.1*dataset_size)
            test_len = dataset_size - train_len - val_len
            train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
            x_val = dataset.tensors[0][val_data.indices, :]
            y_val = dataset.tensors[1][val_data.indices]
            x_test = dataset.tensors[0][test_data.indices, :]
            y_test = dataset.tensors[1][test_data.indices]

            # Setting device
            print(f"Training {name} Classifier...")
            start_time = time.time()

            if method == 'Psi':
                # Network structure
                l1_weight = 1e-6
                hidden_neurons = [10]
                lr_psi = 1e-3
                fan_in = 5
                print("l1 weight", l1_weight)
                print("hidden neurons", hidden_neurons)
                print("lr_psi", lr_psi)
                print("fan_in", fan_in)

                model = PsiNetwork(n_clusters, n_features, hidden_neurons, loss,
                                   l1_weight, name=name, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
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

            elif method == 'General':
                # Network structure
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
                formulas, exp_predictions, exp_complexities = [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_val, y_val, i, top_k_explanations=top_k_explanations,
                                                           concept_names=concept_names, simplify=simplify)
                    _, exp_prediction = test_explanation(formula, i, x_test, y_test,
                                                         metric=F1Score(), concept_names=concept_names)
                    exp_prediction = torch.as_tensor(exp_prediction)
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula)
                    exp_predictions.append(exp_prediction)
                    exp_complexities.append(explanation_complexity)
                    print(f"Formula {i}: {formula}")

            elif method == 'Relu':
                # Network structure
                l1_weight = 1e-5
                hidden_neurons = [50, 30]
                model = XReluNN(n_classes=n_clusters, n_features=n_features, name=name,
                                hidden_neurons=hidden_neurons, loss=loss, l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                formulas, exp_predictions, exp_complexities = [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_val, y_val, i, simplify=simplify,
                                                           top_k_explanations=top_k_explanations,
                                                           concept_names=concept_names)
                    _, exp_prediction = test_explanation(formula, i, x_test, y_test,
                                                         metric=F1Score(), concept_names=concept_names)
                    exp_prediction = torch.as_tensor(exp_prediction)
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula)
                    exp_predictions.append(exp_prediction)
                    exp_complexities.append(explanation_complexity)
                    print(f"Formula {i}: {formula}")

            else:
                raise NotImplementedError(f"{method} not implemented")

            elapsed_time = time.time() - start_time
            outputs, labels = model.predict(test_data, device=device)
            accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
            outputs = outputs.argmax(dim=1)
            exp_predictions = torch.stack(exp_predictions, dim=1)
            exp_accuracy = accuracy_score(exp_predictions, labels, metric)
            exp_fidelity = fidelity(exp_predictions, outputs, metric)
            exp_complexity = np.mean(exp_complexities)
            print("Test model accuracy", accuracy)
            print("Explanation accuracy", exp_accuracy)
            print("Explanation fidelity", exp_fidelity)
            print("Explanation complexity", exp_complexity)
            print("Elapsed time", elapsed_time)

            methods.append(method)
            splits.append(seed)
            explanations.append(formulas[0])
            model_accuracies.append(accuracy)
            explanation_accuracies.append(exp_accuracy)
            explanation_fidelities.append(exp_fidelity)
            elapsed_times.append(elapsed_time)
            explanation_complexities.append(exp_complexity)

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
