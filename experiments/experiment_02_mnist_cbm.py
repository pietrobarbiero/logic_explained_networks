if __name__ == "__main__":

    # %%

    import sys

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    import deep_logic as dl
    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.models.tree import XDecisionTreeClassifier
    from deep_logic.models.brl import XBRLClassifier
    from deep_logic.models.logistic_regression import XLogisticRegressionClassifier
    from deep_logic.utils.base import set_seed, ClassifierNotTrainedError
    from deep_logic.utils.metrics import F1Score
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.utils.datasets import ConceptToTaskDataset
    from deep_logic.utils.data import get_splits_train_val_test
    from deep_logic.logic.base import test_multi_class_explanation, test_explanation
    from data import MNIST
    from experiments.MNIST.concept_extractor_mnist import concept_extractor_mnist

    results_dir = 'results/mnist'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    dataset_root = "../data/MNIST_EVEN_ODD/"
    dataset_root

    # %% md
    ## Define loss, metrics and saved metrics
    # %%

    loss = torch.nn.CrossEntropyLoss()
    metric = F1Score()

    method_list = ['Psi', 'Relu', 'General', 'LogisticRegression', 'DTree', 'BRL']

    #%% md
    ## Loading MNIST data
    #%%

    if not os.path.isfile(os.path.join(dataset_root, f"{MNIST}_predictions.npy")):
        concept_extractor_mnist(dataset_root)
    else:
        print("Concepts already extracted")
    dataset = ConceptToTaskDataset(dataset_root, dataset_name=MNIST, predictions=True)
    concept_names = dataset.attribute_names
    print("Concept names", concept_names)
    n_features = dataset.n_attributes
    print("Number of features", n_features)
    class_names = dataset.classes
    print("Class names", class_names)
    n_classes = dataset.n_classes
    print("Number of classes", n_classes)

    #%% md
    ## Training Hyperparameters
    #%%

    epochs = 20
    l_r = 0.001
    lr_scheduler = True
    top_k_explanations = 5
    simplify = True
    seeds = [*range(1)]
    print("Seeds", seeds)
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

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
            x_test = torch.tensor(dataset.attributes[test_data.indices])
            y_test = torch.tensor(dataset.targets[test_data.indices])
            print(train_data.indices)

            # Setting device
            print(f"Training {name} Classifier...")

            if method == 'BRL':
                model = XBRLClassifier(name=name, n_classes=n_classes, n_features=n_features,
                                       feature_names=concept_names, class_names=dataset.classes, discretize=True)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except ClassifierNotTrainedError:
                    results = model.fit(val_data, metric=metric, save=True, verbose=False)
                    print(results)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, times, exp_accuracies, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula, elapsed_time = model.get_global_explanation(i, concept_names,
                                                                         return_time=True)
                    exp_accuracy, y_formula = test_explanation(formula, i, x_test, y_test,
                                                               metric=metric, concept_names=concept_names)
                    explanation_complexity = dl.logic.complexity(formula)
                    formulas.append(formula), times.append(elapsed_time)
                    exp_accuracies.append(exp_accuracy), exp_complexities.append(explanation_complexity)
                    print(f"{class_to_explain} <-> {formula}")
                    print("Elapsed time", elapsed_time)
                    print("Explanation accuracy", exp_accuracy)
                    print("Explanation complexity", explanation_complexity)

            elif method == 'DTree':
                model = XDecisionTreeClassifier(name=name, n_classes=n_classes, n_features=n_features)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except ClassifierNotTrainedError:
                    results = model.fit(train_data, val_data, metric=metric, save=True)
                    print(results)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, times, exp_accuracies, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula, elapsed_time = model.get_global_explanation(i, concept_names,
                                                                         return_time=True)
                    exp_accuracy = accuracy
                    explanation_complexity = dl.logic.complexity(formula)
                    formulas.append(formula), times.append(elapsed_time)
                    exp_accuracies.append(exp_accuracy), exp_complexities.append(explanation_complexity)
                    print(f"{class_to_explain} <-> {formula}")
                    print("Elapsed time", elapsed_time)
                    print("Explanation accuracy", exp_accuracy)
                    print("Explanation complexity", explanation_complexity)

            elif method == 'Psi':
                # Network structures
                l1_weight = 1e-2
                hidden_neurons = []
                fan_in = 5
                lr_psi = 0.01
                model = PsiNetwork(n_classes, n_features, hidden_neurons, loss,
                                   l1_weight, name=name, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except ClassifierNotTrainedError:
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                    print(results)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, times, exp_accuracies, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula, elapsed_time = model.get_global_explanation(i, concept_names,
                                                                         simplify=True, return_time=True)
                    exp_accuracy, y_formula = test_multi_class_explanation(formula, i, x_test, y_test,
                                                                           metric=metric, concept_names=concept_names)
                    explanation_complexity = dl.logic.complexity(formula)
                    formulas.append(formula), times.append(elapsed_time), exp_accuracies.append(exp_accuracy)
                    exp_complexities.append(explanation_complexity)
                    print(f"{class_to_explain} <-> {formula}")
                    print("Elapsed time", elapsed_time)
                    print("Explanation accuracy", exp_accuracy)
                    print("Explanation complexity", explanation_complexity)

            elif method == 'Relu':
                x_val = torch.tensor(dataset.attributes[val_data.indices])
                y_val = torch.tensor(dataset.targets[val_data.indices])
                # Network structures
                l1_weight = 1e-2
                hidden_neurons = [20, 10]
                model = XReluNN(n_classes=n_classes, n_features=n_features, name=name,
                                hidden_neurons=hidden_neurons, loss=loss, l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except ClassifierNotTrainedError:
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                    print(results)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, times, exp_accuracies, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula, elapsed_time = model.get_global_explanation(x_val, y_val, i,
                                                                         topk_explanations=top_k_explanations,
                                                                         concept_names=concept_names,
                                                                         simplify=True, return_time=True)
                    exp_accuracy, y_formula = test_explanation(formula, i, x_test, y_test,
                                                               metric=metric, concept_names=concept_names)
                    explanation_complexity = dl.logic.complexity(formula)
                    formulas.append(formula), times.append(elapsed_time), exp_accuracies.append(exp_accuracy)
                    exp_complexities.append(explanation_complexity)
                    print(f"{class_to_explain} <-> {formula}")
                    print("Elapsed time", elapsed_time)
                    print("Explanation accuracy", exp_accuracy)
                    print("Explanation complexity", explanation_complexity)

            elif method == 'General':
                x_val = torch.tensor(dataset.attributes[val_data.indices])
                y_val = torch.tensor(dataset.targets[val_data.indices])
                # Network structures
                l1_weight = 1e-1
                hidden_neurons = [10, 5]
                model = XGeneralNN(n_classes=n_classes, n_features=n_features, hidden_neurons=hidden_neurons,
                                   loss=loss, name=name, l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except ClassifierNotTrainedError:
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                                        lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                    print(results)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, times, exp_accuracies, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula, elapsed_time = model.get_global_explanation(x_val, y_val, i, simplify=True,
                                                                         topk_explanations=top_k_explanations,
                                                                         concept_names=concept_names, return_time=True)
                    exp_accuracy, y_formula = test_multi_class_explanation(formula, i, x_test, y_test,
                                                                           metric=metric, concept_names=concept_names)
                    explanation_complexity = dl.logic.complexity(formula)
                    formulas.append(formula), times.append(elapsed_time), exp_accuracies.append(exp_accuracy)
                    exp_complexities.append(explanation_complexity)
                    print(f"{class_to_explain} <-> {formula}")
                    print("Elapsed time", elapsed_time)
                    print("Explanation accuracy", exp_accuracy)
                    print("Explanation complexity", explanation_complexity)

            elif method == 'LogisticRegression':
                set_seed(seed)
                model = XLogisticRegressionClassifier(name=name, n_classes=n_classes, n_features=n_features, loss=loss)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except ClassifierNotTrainedError:
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                                        lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                    print(results)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, times, exp_accuracies, exp_complexities = [""], [0], [0], [0]
            else:
                raise NotImplementedError(f"{method} not implemented")

            methods.append(method)
            splits.append(seed)
            explanations.append(formulas[0])
            model_accuracies.append(accuracy)
            explanation_accuracies.append(np.mean(exp_accuracies))
            elapsed_times.append(np.mean(times))
            explanation_complexities.append(np.mean(exp_complexities))

        explanation_consistency = dl.logic.formula_consistency(explanations)
        print(f'Consistency of explanations: {explanation_consistency:.4f}')

        results = pd.DataFrame({
            'method': methods,
            'split': splits,
            'explanation': explanations,
            'model_accuracy': model_accuracies,
            'explanation_accuracy': explanation_accuracies,
            'explanation_complexity': explanation_complexities,
            'explanation_consistency': explanation_consistency,
            'elapsed_time': elapsed_times,
        })
        results.to_csv(os.path.join(results_dir, f'results_{method}.csv'))
        print(results)

    #%% md
    ##Summary
    #%%

    cols = ['model_accuracy', 'explanation_accuracy', 'explanation_complexity', 'elapsed_time',
            'explanation_consistency']
    mean_cols = [f'{c}_mean' for c in cols]
    sem_cols = [f'{c}_sem' for c in cols]

    # general
    results_general = pd.read_csv(os.path.join(results_dir, "results_general.csv"))
    # results_general = results[results['method'] == "General"]
    df_mean = results_general[cols].mean()
    df_sem = results_general[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_pruning = pd.concat([df_mean, df_sem])
    summary_pruning.name = 'general'

    # lime
    # results_brl = results[results['method'] == "BRL"]
    results_brl = pd.read_csv(os.path.join(results_dir, "results_brl.csv"))
    df_mean = results_brl[cols].mean()
    df_sem = results_brl[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_brl = pd.concat([df_mean, df_sem])
    summary_brl.name = 'brl'

    # relu
    # results_relu = results[results['method'] == "Relu"]
    results_relu = pd.read_csv(os.path.join(results_dir, "results_relu.csv"))
    df_mean = results_relu[cols].mean()
    df_sem = results_relu[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_weights = pd.concat([df_mean, df_sem])
    summary_weights.name = 'relu'

    # psi
    # results_psi = results[results['method'] == "Psi"]
    results_psi = pd.read_csv(os.path.join(results_dir, "results_psi.csv"))
    df_mean = results_psi[cols].mean()
    df_sem = results_psi[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_psi = pd.concat([df_mean, df_sem])
    summary_psi.name = 'psi'

    # tree
    # results_tree = results[results['method'] == "Tree"]
    results_tree = pd.read_csv(os.path.join(results_dir, "results_tree.csv"))
    df_mean = results_tree[cols].mean()
    df_sem = results_tree[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_tree = pd.concat([df_mean, df_sem])
    summary_tree.name = 'tree'

    summary = pd.concat([summary_pruning,
                         summary_brl,
                         summary_weights,
                         summary_psi,
                         summary_tree], axis=1).T
    summary.columns = mean_cols + sem_cols
    print(summary)

    # %%

    summary.to_csv(os.path.join(results_dir, 'summary.csv'))
