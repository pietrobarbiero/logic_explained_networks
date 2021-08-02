
if __name__ == "__main__":

    #%%

    import sys
    import time
    import concurrent

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    from torch.nn import CrossEntropyLoss
    from tqdm import trange
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from lens.models.relu_nn import XReluNN
    from lens.models.psi_nn import PsiNetwork
    from lens.models.tree import XDecisionTreeClassifier
    from lens.models.brl import XBRLClassifier
    from lens.models.logistic_regression import XLogisticRegressionClassifier
    from lens.models.deep_red import XDeepRedClassifier
    from lens.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
    from lens.utils.metrics import Accuracy, F1Score
    from lens.models.general_nn import XGeneralNN
    from lens.utils.datasets import StructuredDataset
    from lens.logic.base import test_explanation
    from lens.logic.metrics import complexity, fidelity, formula_consistency
    from data import MIMIC
    from data.load_structured_datasets import load_mimic

    results_dir = 'results/mimic'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #%% md
    ## Loading MIMIC data
    #%%

    dataset_root = "../data/"
    dataset_name = MIMIC
    print(dataset_root)
    x, y, concept_names, class_names = load_mimic(dataset_root)
    y = y.argmax(dim=1)
    dataset = StructuredDataset(x, y, dataset_name=dataset_name, feature_names=concept_names, class_names=class_names)
    n_features = x.shape[1]
    n_classes = len(class_names)
    print("Number of features", n_features)
    print("Concept names", concept_names)
    print("Class names", class_names)

    #%% md
    ## Define loss, metrics and methods
    #%%

    loss = CrossEntropyLoss()
    metric = Accuracy()
    expl_metric = F1Score()
    method_list = ['DTree', 'BRL', 'Psi', 'Relu', 'General']  # 'DeepRed']
    print("Methods", method_list)

    #%% md
    ## Training
    #%%

    epochs = 1000
    n_processes = 1
    timeout = 6 * 60 * 60  # 1 h timeout
    l_r = 1e-4
    lr_scheduler = False
    top_k_explanations = 20
    simplify = True
    seeds = [*range(10)]  # [*range(5)]
    print("Seeds", seeds)
    device = torch.device("cpu")
    print("Device", device)

    for method in method_list:

        methods = []
        splits = []
        model_explanations = []
        model_accuracies = []
        explanation_accuracies = []
        elapsed_times = []
        explanation_fidelities = []
        explanation_complexities = []

        skf = StratifiedKFold(n_splits=len(seeds), shuffle=True, random_state=0)

        for seed, (trainval_index, test_index) in enumerate(skf.split(x.numpy(), y.numpy())):
            if seed >= 5:
                break
            set_seed(seed)
            x_trainval, y_trainval = x[trainval_index], y[trainval_index]
            x_test, y_test = x[test_index], y[test_index]
            x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.3, random_state=0)
            train_data = StructuredDataset(x_train, y_train, dataset_name, concept_names, class_names)
            val_data = StructuredDataset(x_val, y_val, dataset_name, concept_names, class_names)
            test_data = StructuredDataset(x_test, y_test, dataset_name, concept_names, class_names)

            name = os.path.join(results_dir, f"{method}_{seed}")

            # Setting device
            print(f"Training {name} classifier...")
            start_time = time.time()

            if method == 'DTree':
                model = XDecisionTreeClassifier(name=name, n_classes=n_classes,
                                                n_features=n_features, max_depth=5)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(train_data, val_data, metric=metric, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes, desc=f"{method} extracting explanations"):
                    explanation = model.get_global_explanation(i, concept_names)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test, metric=expl_metric,
                                                                     concept_names=concept_names, inequalities=True)
                    exp_fidelity = 100
                    explanation_complexity = complexity(explanation)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'BRL':
                train_sample_rate = 1.0
                model = XBRLClassifier(name=name, n_classes=n_classes, n_features=n_features, n_processes=n_processes,
                                       feature_names=concept_names, class_names=class_names, discretize=True)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(train_data, metric=metric, train_sample_rate=train_sample_rate, verbose=False, eval=False)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes, desc=f"{method} extracting explanations"):
                    explanation = model.get_global_explanation(i, concept_names)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test, metric=expl_metric,
                                                                     concept_names=concept_names)
                    exp_fidelity = 100
                    explanation_complexity = complexity(explanation, to_dnf=True)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'DeepRed':
                train_sample_rate = 1.0
                model = XDeepRedClassifier(n_classes, n_features, name=name)
                model.prepare_data(dataset, dataset_name, seed, trainval_index, test_index, train_sample_rate)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(epochs=epochs, seed=seed, metric=metric)
                outputs, labels = model.predict(train=False, device=device)
                accuracy = model.evaluate(train=False, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                print("Extracting rules...")
                t = time.time()
                for i in trange(n_classes, desc=f"{method} extracting explanations"):
                    explanation = model.get_global_explanation(i, concept_names, simplify=simplify)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test,
                                                                     metric=expl_metric,
                                                                     concept_names=concept_names, inequalities=True)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)
                    print(f"{i + 1}/{n_classes} Rules extracted. Time {time.time() - t}")

            elif method == 'Psi':
                # Network structures
                l1_weight = 1e-4
                hidden_neurons = [10, 5]
                fan_in = 3
                lr_psi = 1e-2
                print("L1 weight", l1_weight)
                print("Hidden neurons", hidden_neurons)
                print("Fan in", fan_in)
                name = os.path.join(results_dir, f"{method}_{seed}_{l1_weight}_{hidden_neurons}_{fan_in}_{lr_psi}")
                model = PsiNetwork(n_classes, n_features, hidden_neurons, loss, l1_weight, name=name, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                              metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes, desc=f"{method} extracting explanations"):
                    explanation = model.get_global_explanation(i, concept_names, simplify=simplify, x_train=x_train)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test,
                                                                     metric=expl_metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation, to_dnf=True)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'General':
                # Network structures
                l1_weight = 1e-4
                hidden_neurons = [100, 30, 10]
                fan_in = 5
                top_k_explanations = None
                name = os.path.join(results_dir, f"{method}_{seed}_{l1_weight}_{hidden_neurons}_{fan_in}")
                model = XGeneralNN(n_classes=n_classes, n_features=n_features, hidden_neurons=hidden_neurons,
                                   loss=loss, name=name, l1_weight=l1_weight, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                              lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes, desc=f"{method} extracting explanations"):
                    explanation = model.get_global_explanation(x_train, y_train, i, top_k_explanations=top_k_explanations,
                                                               concept_names=concept_names, simplify=simplify,
                                                               metric=expl_metric, x_val=x_val, y_val=y_val)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test,
                                                                     metric=expl_metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'Relu':
                # Network structures
                l1_weight = 1e-4
                hidden_neurons = [100, 50, 30, 10]
                dropout_rate = 0.3
                print("l1 weight", l1_weight)
                print("hidden neurons", hidden_neurons)
                model = XReluNN(n_classes=n_classes, n_features=n_features, name=name, dropout_rate=dropout_rate,
                                hidden_neurons=hidden_neurons, loss=loss, l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                              metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes, desc=f"{method} extracting explanations"):
                    explanation = model.get_global_explanation(x_train, y_train, i,
                                                               top_k_explanations=top_k_explanations,
                                                               concept_names=concept_names,
                                                               metric=expl_metric, x_val=x_val, y_val=y_val)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, x_test, y_test,
                                                                     metric=expl_metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'LogisticRegression':
                set_seed(seed)
                model = XLogisticRegressionClassifier(name=name, n_classes=n_classes, n_features=n_features, loss=loss)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                              lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                accuracy = model.evaluate(test_data, metric=metric)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [""], [0], [0], [0]
            else:
                raise NotImplementedError(f"{method} not implemented")

            if model.time is None:
                elapsed_time = time.time() - start_time
                # In BRL the training is parallelized to speed up operation
                if method == "BRL":
                    elapsed_time = elapsed_time * n_processes
                model.time = elapsed_time
                # To save the elapsed time and the explanations
                model.save(device)
            else:
                elapsed_time = model.time

            # Restore original folder
            if method == "DeepRed":
                model.finish()

            methods.append(method)
            splits.append(seed)
            model_explanations.append(explanations[0])
            model_accuracies.append(accuracy)
            elapsed_times.append(elapsed_time)
            explanation_accuracies.append(np.mean(exp_accuracies))
            explanation_fidelities.append(np.mean(exp_fidelities))
            explanation_complexities.append(np.mean(exp_complexities))
            print("Test model accuracy", accuracy)
            print("Explanation time", elapsed_time)
            print("Explanation accuracy mean", np.mean(exp_accuracies))
            print("Explanation fidelity mean", np.mean(exp_fidelities))
            print("Explanation complexity mean", np.mean(exp_complexities))

        explanation_consistency = formula_consistency(model_explanations)
        print(f'Consistency of explanations: {explanation_consistency:.4f}')

        results = pd.DataFrame({
            'method': methods,
            'split': splits,
            'explanation': model_explanations,
            'model_accuracy': model_accuracies,
            'explanation_accuracy': explanation_accuracies,
            'explanation_fidelity': explanation_fidelities,
            'explanation_complexity': explanation_complexities,
            'explanation_consistency': [explanation_consistency] * len(splits),
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

    results_df = {}
    summaries = {}
    for m in method_list:
        results_df[m] = pd.read_csv(os.path.join(results_dir, f"results_{m}.csv"))
        df_mean = results_df[m][cols].mean()
        df_sem = results_df[m][cols].sem()
        df_mean.columns = mean_cols
        df_sem.columns = sem_cols
        summaries[m] = pd.concat([df_mean, df_sem])
        summaries[m].name = m

    results_df = pd.concat([results_df[method] for method in method_list])
    results_df.to_csv(os.path.join(results_dir, f'results.csv'))

    summary = pd.concat([summaries[method] for method in method_list], axis=1).T
    summary.columns = mean_cols + sem_cols
    summary.to_csv(os.path.join(results_dir, 'summary.csv'))
    print(summary)

    #%%
