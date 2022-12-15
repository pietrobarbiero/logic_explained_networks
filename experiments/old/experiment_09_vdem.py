
if __name__ == "__main__":

    #%%

    import sys
    import time

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from tqdm import trange

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

    from lens.models.relu_nn import XReluNN
    from lens.models.psi_nn import XPsiNetwork
    from lens.models.tree import XDecisionTreeClassifier
    from lens.models.brl import XBRLClassifier
    from lens.models.deep_red import XDeepRedClassifier
    from lens.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
    from lens.utils.metrics import Accuracy, F1Score
    from lens.models.mu_nn import XMuNN
    from lens.utils.datasets import StructuredDataset
    from lens.logic.base import test_explanation
    from lens.logic.metrics import complexity, fidelity, formula_consistency
    from data import VDEM
    from data.load_structured_datasets import load_vDem

    # n_sample = 100
    results_dir = f'results/vDem'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #%% md
    ## Loading VDEM data
    #%%

    dataset_root = "../data/"
    dataset_name = VDEM
    print(dataset_root)
    print(results_dir)
    x, c, y, feature_names, concept_names, class_names = load_vDem(dataset_root)
    y = y.argmax(dim=1)
    n_features = x.shape[1]
    n_concepts = c.shape[1]
    n_classes = len(class_names)
    dataset_low = StructuredDataset(x, c, dataset_name=dataset_name, feature_names=feature_names, class_names=concept_names)
    print("Number of features", n_features)
    print("Number of concepts", n_concepts)
    print("Feature names", feature_names)
    print("Concept names", concept_names)
    print("Class names", class_names)

    #%% md
    ## Define loss, metrics and methods
    #%%
    loss_low = BCEWithLogitsLoss()
    loss_high = CrossEntropyLoss()
    metric = Accuracy()
    expl_metric = F1Score()
    method_list = ['DTree', 'BRL', 'Psi', 'Relu', 'General']  # 'DeepRed']
    print("Methods", method_list)

    #%% md
    ## Training
    #%%

    epochs = 1000
    n_processes = 4
    timeout = 60 * 60  # 1 h timeout
    l_r = 1e-3
    lr_scheduler = False
    top_k_explanations = None
    simplify = True
    seeds = [*range(5)]
    print("Seeds", seeds)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
            set_seed(seed)
            x_trainval, c_trainval, y_trainval = x[trainval_index], c[trainval_index], y[trainval_index]
            x_test, c_test, y_test = x[test_index], c[test_index], y[test_index]
            x_train, x_val, c_train, c_val, y_train, y_val = train_test_split(x_trainval, c_trainval, y_trainval,
                                                                              test_size=0.3, random_state=0)
            train_data_low = StructuredDataset(x_train, c_train, dataset_name, feature_names, concept_names)
            val_data_low = StructuredDataset(x_val, c_val, dataset_name, feature_names, concept_names)
            test_data_low = StructuredDataset(x_test, c_test, dataset_name, feature_names, concept_names)
            data_low = StructuredDataset(x, c, dataset_name, feature_names, concept_names)

            name_low = os.path.join(results_dir, f"{method}_{seed}_low")
            name_high = os.path.join(results_dir, f"{method}_{seed}_high")

            # Setting device
            print(f"Training {name_low} classifier...")
            start_time = time.time()

            if method == 'DTree':
                model_low = XDecisionTreeClassifier(name=name_low, n_classes=n_concepts,
                                                    n_features=n_features, max_depth=5)
                try:
                    model_low.load(device)
                    print(f"Model {name_low} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(train_data_low, val_data_low, metric=metric, save=True)
                c_predicted_train, _ = model_low.predict(train_data_low, device=device)
                c_predicted_val, _ = model_low.predict(val_data_low, device=device)
                c_predicted_test, _ = model_low.predict(test_data_low, device=device)
                accuracy_low = model_low.evaluate(test_data_low, metric=metric)
                train_data_high = StructuredDataset(c_predicted_train, y_train, dataset_name, feature_names, concept_names)
                val_data_high = StructuredDataset(c_predicted_val, y_val, dataset_name, feature_names, concept_names)
                test_data_high = StructuredDataset(c_predicted_test, y_test, dataset_name, feature_names, concept_names)
                model_high = XDecisionTreeClassifier(name=name_high, n_classes=n_classes, n_features=n_concepts, max_depth=5)
                try:
                    model_high.load(device)
                    print(f"Model {name_high} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_high.fit(train_data_high, val_data_high, metric=metric, save=True)
                outputs, labels = model_high.predict(test_data_high, device=device)
                accuracy = model_high.evaluate(test_data_high, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes):
                    explanation = model_high.get_global_explanation(i, concept_names)
                    class_output = torch.as_tensor((outputs > 0.5) == i)
                    class_label = torch.as_tensor(labels == i)
                    exp_fidelity = 100
                    exp_accuracy = expl_metric(class_output, class_label)
                    explanation_complexity = complexity(explanation)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'BRL':
                train_sample_rate = 1.0
                model_low = XBRLClassifier(name=name_low, n_classes=n_concepts, n_features=n_features,
                                           n_processes=n_processes, feature_names=feature_names, class_names=concept_names)
                try:
                    model_low.load(device)
                    print(f"Model {name_low} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(train_data_low, train_sample_rate=train_sample_rate,
                                  verbose=True, eval=False)
                c_predicted, _ = model_low.predict(data_low, device=device)
                c_predicted_train, c_predicted_test = c_predicted[trainval_index], c_predicted[test_index]
                accuracy_low = model_low.evaluate(test_data_low, metric=metric, outputs=c_predicted_test, labels=c_test)
                train_data_high = StructuredDataset(c_predicted_train, y_trainval, dataset_name, feature_names, concept_names)
                test_data_high = StructuredDataset(c_predicted_test, y_test, dataset_name, feature_names, concept_names)
                model_high = XBRLClassifier(name=name_high, n_classes=n_classes, n_features=n_concepts,
                                            n_processes=n_processes, feature_names=concept_names, class_names=class_names)
                try:
                    model_high.load(device)
                    print(f"Model {name_high} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_high.fit(train_data_high, train_sample_rate=train_sample_rate, verbose=True,
                                   eval=False)
                outputs, labels = model_high.predict(test_data_high, device=device)
                accuracy = model_high.evaluate(test_data_high, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes):
                    explanation = model_high.get_global_explanation(i, concept_names)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, c_predicted_test, y_test, metric=expl_metric,
                                                                     concept_names=concept_names)
                    exp_fidelity = 100
                    explanation_complexity = complexity(explanation, to_dnf=True)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'DeepRed':
                train_sample_rate = 0.1
                model_low = XDeepRedClassifier(name=name_low, n_classes=n_concepts, n_features=n_features)
                model_low.prepare_data(dataset_low, dataset_name + "low", seed, trainval_index, test_index, train_sample_rate)
                try:
                    model_low.load(device)
                    print(f"Model {name_low} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(epochs,  train_sample_rate=train_sample_rate, verbose=True, eval=False)
                c_predicted_train, _ = model_low.predict(train=True, device=device)
                c_predicted_test, _ = model_low.predict(train=False, device=device)
                accuracy_low = model_low.evaluate(train=False, outputs=c_predicted_test, labels=c_test, metric=metric)
                model_low.finish()
                c_predicted = torch.vstack((c_predicted_train, c_predicted_test))
                y = torch.vstack((y_train, y_test))
                dataset_high = StructuredDataset(c_predicted, y, dataset_name, feature_names, concept_names)
                model_high = XDeepRedClassifier(n_classes, n_features, name=name_high)
                model_high.prepare_data(dataset_high, dataset_name + "high", seed, trainval_index, test_index, train_sample_rate)
                try:
                    model_high.load(device)
                    print(f"Model {name_high} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(epochs,  train_sample_rate=train_sample_rate, verbose=True, eval=False)
                outputs, labels = model_high.predict(train=False, device=device)
                accuracy = model_high.evaluate(train=False, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                print("Extracting rules...")
                t = time.time()
                for i in trange(n_classes):
                    explanation = model_high.get_global_explanation(i, concept_names, simplify=simplify)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, c_predicted_test, y_test,
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
                print("Learning rate", lr_psi)
                name_low = os.path.join(results_dir, f"{method}_{seed}_{l1_weight}_{hidden_neurons}_{fan_in}_{lr_psi}_low")
                name_high = os.path.join(results_dir, f"{method}_{seed}_{l1_weight}_{hidden_neurons}_{fan_in}_{lr_psi}_high")
                model_low = XPsiNetwork(n_concepts, n_features, hidden_neurons, loss_low, l1_weight, name=name_low,
                                        fan_in=fan_in)
                try:
                    model_low.load(device)
                    print(f"Model {name_low} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(train_data_low, val_data_low, epochs=epochs, l_r=lr_psi,
                                  metric=metric, lr_scheduler=lr_scheduler, device=device, verbose=True)
                c_predicted_train = model_low.predict(train_data_low, device=device)[0].detach().cpu()
                c_predicted_val = model_low.predict(val_data_low, device=device)[0].detach().cpu()
                c_predicted_test = model_low.predict(test_data_low, device=device)[0].detach().cpu()
                accuracy_low = model_low.evaluate(test_data_low, outputs=c_predicted_test, labels=c_test, metric=metric)
                train_data_high = StructuredDataset(c_predicted_train, y_train, dataset_name, feature_names, concept_names)
                val_data_high = StructuredDataset(c_predicted_val, y_val, dataset_name, feature_names, concept_names)
                test_data_high = StructuredDataset(c_predicted_test, y_test, dataset_name, feature_names, concept_names)
                model_high = XPsiNetwork(n_classes, n_concepts, hidden_neurons, loss_high, l1_weight,
                                         name=name_high, fan_in=fan_in)
                try:
                    model_high.load(device)
                    print(f"Model {name_high} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_high.fit(train_data_high, val_data_high, epochs=epochs, l_r=lr_psi,
                                   metric=metric, lr_scheduler=lr_scheduler, device=device, verbose=True)
                outputs, labels = model_high.predict(test_data_high, device=device)
                accuracy = model_high.evaluate(test_data_high, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes):
                    explanation = model_high.get_global_explanation(i, concept_names, simplify=simplify, x_train=c_predicted_train)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, c_predicted_test, y_test,
                                                                     metric=expl_metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation, to_dnf=True)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'General':
                # Network structures
                l1_weight = 1e-3
                hidden_neurons = [100, 30, 10]
                fan_in = 5
                top_k_explanations = None
                name_low = os.path.join(results_dir, f"{method}_{seed}_{l1_weight}_{hidden_neurons}_{fan_in}_low")
                name_high = os.path.join(results_dir, f"{method}_{seed}_{l1_weight}_{hidden_neurons}_{fan_in}_high")

                model_low = XMuNN(n_concepts, n_features, hidden_neurons, fan_in=n_features,
                                  loss=loss_low, name=name_low, l1_weight=l1_weight)
                try:
                    model_low.load(device)
                    print(f"Model {name_low} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(train_data_low, val_data_low, epochs=epochs, l_r=l_r,
                                  metric=metric, lr_scheduler=lr_scheduler, device=device, verbose=True)
                c_predicted_train = model_low.predict(train_data_low, device=device)[0].detach().cpu()
                c_predicted_val = model_low.predict(val_data_low, device=device)[0].detach().cpu()
                c_predicted_test = model_low.predict(test_data_low, device=device)[0].detach().cpu()
                accuracy_low = model_low.evaluate(test_data_low, outputs=c_predicted_test, labels=c_test, metric=metric)
                train_data_high = StructuredDataset(c_predicted_train, y_train, dataset_name, feature_names, concept_names)
                val_data_high = StructuredDataset(c_predicted_val, y_val, dataset_name, feature_names, concept_names)
                test_data_high = StructuredDataset(c_predicted_test, y_test, dataset_name, feature_names, concept_names)

                model_high = XMuNN(n_classes, n_concepts, hidden_neurons, fan_in=fan_in,
                                   loss=loss_high, name=name_high, l1_weight=l1_weight)
                try:
                    model_high.load(device)
                    print(f"Model {name_high} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_high.fit(train_data_high, val_data_high, epochs=epochs, l_r=l_r*1e-1,
                                   metric=metric, lr_scheduler=lr_scheduler, device=device, verbose=True)
                outputs, labels = model_high.predict(test_data_high, device=device)
                accuracy = model_high.evaluate(test_data_high, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes):
                    explanation = model_high.get_global_explanation(c_predicted_train, y_train, i,
                                                                    top_k_explanations=top_k_explanations,
                                                                    concept_names=concept_names, simplify=simplify,
                                                                    metric=expl_metric, x_val=c_predicted_val,
                                                                    y_val=y_val)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, c_predicted_test, y_test,
                                                                     metric=expl_metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation, to_dnf=True)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            elif method == 'Relu':
                # Network structures
                l1_weight = 1e-4
                hidden_neurons = [100, 50, 30, 10]
                dropout_rate = 0.01
                print("l1 weight", l1_weight)
                print("hidden neurons", hidden_neurons)
                model_low = XReluNN(n_classes=n_concepts, n_features=n_features, name=name_low, dropout_rate=dropout_rate,
                                    hidden_neurons=hidden_neurons, loss=loss_low, l1_weight=l1_weight*1e-2)
                try:
                    model_low.load(device)
                    print(f"Model {name_low} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_low.fit(train_data_low, val_data_low, epochs=epochs, l_r=l_r,
                                  metric=metric, lr_scheduler=lr_scheduler, device=device, verbose=True)
                c_predicted_train = model_low.predict(train_data_low, device=device)[0].detach().cpu()
                c_predicted_val = model_low.predict(val_data_low, device=device)[0].detach().cpu()
                c_predicted_test = model_low.predict(test_data_low, device=device)[0].detach().cpu()
                accuracy_low = model_low.evaluate(test_data_low, outputs=c_predicted_test, labels=c_test, metric=metric)
                train_data_high = StructuredDataset(c_predicted_train, y_train, dataset_name, feature_names, concept_names)
                val_data_high = StructuredDataset(c_predicted_val, y_val, dataset_name, feature_names, concept_names)
                test_data_high = StructuredDataset(c_predicted_test, y_test, dataset_name, feature_names, concept_names)
                model_high = XReluNN(n_classes=n_classes, n_features=n_concepts, name=name_high, dropout_rate=dropout_rate,
                                    hidden_neurons=hidden_neurons, loss=loss_high, l1_weight=l1_weight)
                try:
                    model_high.load(device)
                    print(f"Model {name_high} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    model_high.fit(train_data_high, val_data_high, epochs=epochs, l_r=l_r * 1e-1,
                                   metric=metric, lr_scheduler=lr_scheduler, device=device, verbose=True)
                outputs, labels = model_high.predict(test_data_high, device=device)
                accuracy = model_high.evaluate(test_data_high, metric=metric, outputs=outputs, labels=labels)
                explanations, exp_accuracies, exp_fidelities, exp_complexities = [], [], [], []
                for i in trange(n_classes):
                    explanation = model_high.get_global_explanation(c_predicted_train, y_train, i,
                                                                    top_k_explanations=top_k_explanations,
                                                                    concept_names=concept_names, simplify=simplify,
                                                                    metric=expl_metric, x_val=c_predicted_val, y_val=y_val)
                    exp_accuracy, exp_predictions = test_explanation(explanation, i, c_predicted_test, y_test,
                                                                     metric=expl_metric, concept_names=concept_names)
                    exp_predictions = torch.as_tensor(exp_predictions)
                    class_output = torch.as_tensor(outputs.argmax(dim=1) == i)
                    exp_fidelity = fidelity(exp_predictions, class_output, expl_metric)
                    explanation_complexity = complexity(explanation, to_dnf=True)
                    explanations.append(explanation), exp_accuracies.append(exp_accuracy)
                    exp_fidelities.append(exp_fidelity), exp_complexities.append(explanation_complexity)

            else:
                raise NotImplementedError(f"{method} not implemented")

            if model_high.time is None:
                elapsed_time = time.time() - start_time
                # In DeepRed and BRL the training is parallelized to speed up operation
                if method == "DeepRed" or method == "BRL":
                    elapsed_time = elapsed_time * n_processes
                model_high.time = elapsed_time
                # To save the elapsed time and the explanations
                model_high.save(device)
            else:
                elapsed_time = model_high.time

            # To restore the original folder
            if method == "DeepRed":
                model_high.finish()

            methods.append(method)
            splits.append(seed)
            model_explanations.append(explanations[0])
            model_accuracies.append(accuracy)
            elapsed_times.append(elapsed_time)
            explanation_accuracies.append(np.mean(exp_accuracies))
            explanation_fidelities.append(np.mean(exp_fidelities))
            explanation_complexities.append(np.mean(exp_complexities))
            print("Test model low accuracy", accuracy_low)
            print("Test model high accuracy", accuracy)
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
            'explanation_consistency': [explanation_consistency] * len(seeds),
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
    summary.to_csv(os.path.join(results_dir, 'summary_2.csv'))
    print(summary)

    #%%
