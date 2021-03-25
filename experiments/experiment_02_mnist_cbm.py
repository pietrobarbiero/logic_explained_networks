
if __name__ == "__main__":

    # %%

    import sys
    import time

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.models.tree import XDecisionTreeClassifier
    from deep_logic.models.brl import XBRLClassifier
    from deep_logic.models.logistic_regression import XLogisticRegressionClassifier
    from deep_logic.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
    from deep_logic.utils.metrics import Accuracy
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.utils.datasets import ConceptToTaskDataset
    from deep_logic.utils.data import get_splits_train_val_test
    from deep_logic.logic.base import test_explanation
    from deep_logic.logic.metrics import complexity, fidelity, formula_consistency
    from data import MNIST
    from data.download_mnist import download_mnist
    from experiments.MNIST.concept_extractor_mnist import concept_extractor_mnist

    results_dir = 'results/mnist'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # %% md
    ## Loading MNIST data
    # %%

    dataset_root = "../data/MNIST_EVEN_ODD/"
    print(dataset_root)
    if not os.path.isdir(dataset_root):
        download_mnist(dataset_root)
    else:
        print("Dataset already downloaded")

    #%% md
    ## Extracting concepts
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
    ## Define loss, metrics and methods
    #%%

    loss = torch.nn.CrossEntropyLoss()
    metric = Accuracy()
    method_list = ['General', 'Relu', 'Psi', 'DTree', 'BRL', ]  # TODO: 'LogisticRegression',
    print(method_list)

    #%% md
    ## Training
    #%%

    epochs = 200
    l_r = 1e-3
    lr_scheduler = False
    top_k_explanations = 5
    simplify = True
    seeds = [*range(10)]
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
            x_val = torch.tensor(dataset.attributes[val_data.indices])
            y_val = torch.tensor(dataset.targets[val_data.indices])
            x_test = torch.tensor(dataset.attributes[test_data.indices])
            y_test = torch.tensor(dataset.targets[test_data.indices])
            print(train_data.indices)

            # Setting device
            print(f"Training {name} classifier...")
            start_time = time.time()

            if method == 'BRL':
                model = XBRLClassifier(name=name, n_classes=n_classes, n_features=n_features,
                                       feature_names=concept_names, class_names=dataset.classes, discretize=True)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, metric=metric, save=True, verbose=False, eval=False)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(i, concept_names)
                    exp_accuracy = accuracy
                    exp_fidelity = 100
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)
                    # print(f"{class_to_explain} <-> {formula}")
                    # print("Explanation accuracy", exp_accuracy)
                    # print("Explanation complexity", explanation_complexity)

            elif method == 'DTree':
                model = XDecisionTreeClassifier(name=name, n_classes=n_classes, n_features=n_features)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, metric=metric, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(i, concept_names)
                    exp_accuracy = accuracy
                    exp_fidelity = 100
                    explanation_complexity = complexity(formula)
                    formulas.append(formula), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)
                    # print(f"{class_to_explain} <-> {formula}")
                    # print("Explanation accuracy", exp_accuracy)
                    # print("Explanation complexity", explanation_complexity)

            elif method == 'Psi':
                # Network structures
                l1_weight = 1e-3
                hidden_neurons = []
                fan_in = 5
                lr_psi = 1e-3
                lr_scheduler_psi = True
                print("Lr", lr_psi)
                print("lr_scheduler", lr_scheduler_psi)
                print("L1 weight", l1_weight)
                print("Hidden neurons", hidden_neurons)
                print("Fan in", fan_in)
                model = PsiNetwork(n_classes, n_features, hidden_neurons, loss,
                                   l1_weight, name=name, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler_psi, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(i, concept_names, simplify=simplify)
                    exp_accuracy, exp_predictions = test_explanation(formula, i, x_test, y_test,
                                                                     metric=metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = outputs.argmax(dim=1) == i
                    exp_fidelity = fidelity(exp_predictions, class_output, metric)
                    explanation_complexity = complexity(formula, to_dnf=True)
                    formulas.append(formula), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)
                    # print(f"{class_to_explain} <-> {formula}")
                    # print("Explanation accuracy", exp_accuracy)
                    # print("Explanation complexity", explanation_complexity)

            elif method == 'General':
                # Network structures
                l1_weight = 1e-2
                hidden_neurons = [10]
                model = XGeneralNN(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons,
                                   loss=torch.nn.BCEWithLogitsLoss(), name=name, l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                                        lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_val, y_val, i, simplify=simplify,
                                                           topk_explanations=top_k_explanations,
                                                           concept_names=concept_names)
                    exp_accuracy, exp_predictions = test_explanation(formula, i, x_test, y_test,
                                                                     metric=metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.tensor((outputs > 0.5) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, metric)
                    explanation_complexity = complexity(formula)
                    formulas.append(formula), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)
                    # print(f"{class_to_explain} <-> {formula}")
                    # print("Explanation accuracy", exp_accuracy)
                    # print("Explanation complexity", explanation_complexity)

            elif method == 'Relu':
                # Network structures
                l1_weight = 1e-3
                print("l1 weight", l1_weight)
                hidden_neurons = [30, 10]
                model = XReluNN(n_classes=1, n_features=n_features, name=name,
                                hidden_neurons=hidden_neurons, loss=torch.nn.BCEWithLogitsLoss(), l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_val, y_val, i,
                                                           topk_explanations=top_k_explanations,
                                                           concept_names=concept_names,
                                                           simplify=simplify)
                    exp_accuracy, exp_predictions = test_explanation(formula, i, x_test, y_test,
                                                                     metric=metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.tensor((outputs > 0.5) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, metric)
                    explanation_complexity = complexity(formula)
                    formulas.append(formula), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)
                    # print(f"{class_to_explain} <-> {formula}")
                    # print("Explanation accuracy", exp_accuracy)
                    # print("Explanation complexity", explanation_complexity)

            elif method == 'LogisticRegression':
                l_r_lr = 1e-1
                set_seed(seed)
                model = XLogisticRegressionClassifier(name=name, n_classes=1, n_features=n_features,
                                                      loss=torch.nn.BCEWithLogitsLoss())
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r_lr, metric=metric,
                                        lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                accuracy = model.evaluate(test_data, metric=metric)
                print("Test model accuracy", accuracy)
                formulas, exp_accuracies, exp_fidelities, exp_complexities = [""], [0], [0], [0]
            else:
                raise NotImplementedError(f"{method} not implemented")

            elapsed_time = time.time() - start_time

            methods.append(method)
            splits.append(seed)
            explanations.append(formulas[0])
            model_accuracies.append(accuracy)
            elapsed_times.append(elapsed_time)
            explanation_accuracies.append(np.mean(exp_accuracies))
            explanation_fidelities.append(np.mean(exp_fidelities))
            explanation_complexities.append(np.mean(exp_complexities))
            print("Explanation time", elapsed_time)
            print("Explanation accuracy mean", np.mean(exp_accuracies))
            print("Explanation fidelity mean", np.mean(exp_fidelities))
            print("Explanation complexity mean", np.mean(exp_complexities))

        explanation_consistency = formula_consistency(explanations)
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

    #%%
