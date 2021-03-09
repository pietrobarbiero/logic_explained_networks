if __name__ == '__main__':

    # %%
    import sys

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    import numpy as np
    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.models.tree import XDecisionTreeClassifier
    from deep_logic.utils.base import set_seed
    from deep_logic.utils.metrics import F1Score
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.utils.datasets import ConceptToTaskDataset
    from deep_logic.utils.data import get_splits_train_val_test
    from deep_logic.logic.base import test_multi_class_explanation

    results_dir = 'results/cub'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # %% md
    ## Define loss, metrics and saved metrics
    # %%
    loss = torch.nn.CrossEntropyLoss()
    metric = F1Score()

    methods = []
    splits = []
    explanations = []
    explanations_inv = []
    model_accuracies = []
    explanation_accuracies = []
    explanation_accuracies_inv = []
    elapsed_times = []
    elapsed_times_inv = []

    # %% md
    ## Loading CUB data
    # %%
    dataset = ConceptToTaskDataset("../data/CUB_200_2011/")
    concept_names = dataset.attribute_names
    print("Concept names", concept_names)
    n_features = dataset.n_attributes
    print("Number of features", n_features)

    # %% md
    ## Training Hyperparameters
    # %%
    epochs = 200
    l_r = 0.001
    lr_scheduler = True
    top_k_explanations = 1
    simplify = True
    seeds = [*range(10)]
    print("Seeds", seeds)

    #%% md
    ## Decision Tree
    #%%

    for seed in seeds:
        print("Seed", seed)
        set_seed(seed)

        train_data, val_data, test_data = get_splits_train_val_test(dataset)
        print(train_data.indices)

        device = torch.device("cpu")

        print("Training Tree Classifier...")
        model = XDecisionTreeClassifier(n_classes=dataset.n_classes, n_features=n_features)

        results = model.fit(train_data, val_data, metric=metric, save=False)
        print(results)

        accuracy = model.evaluate(test_data)
        print("Test model accuracy", accuracy)

        formulas, times = [], []
        for i, class_to_explain in enumerate(dataset.classes):
            formula, elapsed_time = model.get_global_explanation(i, concept_names,
                                                                 return_time=True)
            formulas.append(formula), times.append(elapsed_time)
            print(f"{class_to_explain} <-> {formula}")
            print("Elapsed time", elapsed_time)

        methods.append("Tree")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        model_accuracies.append(accuracy)
        explanation_accuracies.append(accuracy)
        explanation_accuracies_inv.append(accuracy)
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    results_tree = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'model_accuracy': model_accuracies,
        'explanation_accuracy': explanation_accuracies,
        'explanation_accuracy_inv': explanation_accuracies_inv,
        'elapsed_time': elapsed_times,
        'elapsed_time_inv': elapsed_times_inv,
    })
    results_tree.to_csv(os.path.join(results_dir, 'results_tree.csv'))
    print(results_tree)

    #%% md
    ## PSI NN
    #%%

    for seed in seeds:
        print("Seed", seed)
        set_seed(seed)

        train_data, val_data, test_data = get_splits_train_val_test(dataset)
        print(train_data.indices)

        x_test = torch.tensor(dataset.attributes[test_data.indices])
        y_test = torch.tensor(dataset.targets[test_data.indices])

        # Network structures
        l1_weight = 1e-4
        hidden_neurons = [10, 5, 2]
        fan_in = 2
        lr_psi = 0.01

        # Setting device
        device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)

        print("Training Psi NN...")
        model = PsiNetwork(dataset.n_classes, n_features, hidden_neurons, loss,
                           l1_weight, fan_in=fan_in)

        results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_psi, verbose=True,
                            metric=metric, lr_scheduler=lr_scheduler, device=device, save=False)
        print(results)
        accuracy = model.evaluate(test_data, metric=metric)
        print("Test model accuracy", accuracy)

        formulas, times, exp_accuracies = [], [], []
        for i, class_to_explain in enumerate(dataset.classes):
            formula, elapsed_time = model.get_global_explanation(i, concept_names,
                                                                 simplify=True, return_time=True)
            exp_accuracy, _ = test_multi_class_explanation(formula, i, x_test, y_test,
                                                           metric=metric, concept_names=concept_names)
            formulas.append(formula), times.append(elapsed_time), exp_accuracies.append(exp_accuracy)
            print(f"{class_to_explain} <-> {formula}")
            print("Elapsed time", elapsed_time)
            print("Explanation accuracy", exp_accuracy)

        methods.append("Psi")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        model_accuracies.append(accuracy)
        explanation_accuracies.append(np.mean(exp_accuracies))
        explanation_accuracies_inv.append(np.mean(exp_accuracies))
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    results = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'model_accuracy': model_accuracies,
        'explanation_accuracy': explanation_accuracies,
        'explanation_accuracy_inv': explanation_accuracies_inv,
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
        y_val = torch.tensor(dataset.targets[val_data.indices])
        x_test = torch.tensor(dataset.attributes[test_data.indices])
        y_test = torch.tensor(dataset.targets[test_data.indices])

        # Network structures
        l1_weight = 1e-3
        hidden_neurons = [10, 5, 2]

        # Setting device
        device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)

        print("Training General NN...")
        model = XGeneralNN(n_classes=dataset.n_classes, n_features=n_features, hidden_neurons=hidden_neurons,
                           loss=loss, l1_weight=l1_weight)

        results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, metric=metric,
                            lr_scheduler=lr_scheduler, device=device, save=False, verbose=True)
        print(results)
        accuracy = model.evaluate(test_data, metric=metric)
        print("Test model accuracy", accuracy)

        formulas, times, exp_accuracies = [], [], []
        for i, class_to_explain in enumerate(dataset.classes):
            formula, elapsed_time = model.get_global_explanation(x_val, y_val, i, simplify=True,
                                                                 topk_explanations=top_k_explanations,
                                                                 concept_names=concept_names, return_time=True)
            exp_accuracy, _ = test_multi_class_explanation(formula, i, x_test, y_test,
                                                           metric=metric, concept_names=concept_names)
            formulas.append(formula), times.append(elapsed_time), exp_accuracies.append(exp_accuracy)
            print(f"{class_to_explain} <-> {formula}")
            print("Elapsed time", elapsed_time)
            print("Explanation accuracy", exp_accuracy)
        mean_exp_accuracy = np.mean(exp_accuracies)
        print("Mean exp accuracy", mean_exp_accuracy)

        methods.append("General")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        model_accuracies.append(accuracy)
        explanation_accuracies.append(mean_exp_accuracy)
        explanation_accuracies_inv.append(mean_exp_accuracy)
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    results = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'model_accuracy': model_accuracies,
        'explanation_accuracy': explanation_accuracies,
        'explanation_accuracy_inv': explanation_accuracies_inv,
        'elapsed_time': elapsed_times,
        'elapsed_time_inv': elapsed_times_inv,
    })
    results_general = results[results['method'] == "General"]
    results_general.to_csv(os.path.join(results_dir, 'results_general.csv'))
    results.to_csv(os.path.join(results_dir, 'results.csv'))
    print(results)

    #%% md
    ## Relu NN
    #%%

    for seed in seeds:
        print("Seed", seed)
        set_seed(seed)

        train_data, val_data, test_data = get_splits_train_val_test(dataset)
        print(train_data.indices)

        x_val = torch.tensor(dataset.attributes[val_data.indices])
        y_val = torch.tensor(dataset.targets[val_data.indices])
        x_test = torch.tensor(dataset.attributes[test_data.indices])
        y_test = torch.tensor(dataset.targets[test_data.indices])

        # Network structures
        l1_weight = 1e-4
        hidden_neurons = [200, 100]

        # Setting device
        device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
        set_seed(seed)
        print(f"Training Relu NN...")
        model = XReluNN(n_classes=dataset.n_classes, n_features=n_features,
                        hidden_neurons=hidden_neurons, loss=loss, l1_weight=l1_weight)

        results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                            metric=metric, lr_scheduler=lr_scheduler, device=device, save=False)
        print(results)
        accuracy = model.evaluate(test_data)
        print("Test Model accuracy", accuracy)

        formulas, times, exp_accuracies = [], [], []
        for i, class_to_explain in enumerate(dataset.classes):
            formula, elapsed_time = model.get_global_explanation(x_val, y_val, i,
                                                                 topk_explanations=top_k_explanations,
                                                                 concept_names=concept_names,
                                                                 simplify=True, return_time=True)
            exp_accuracy, _ = test_multi_class_explanation(formula, i, x_test, y_test,
                                                           metric=metric, concept_names=concept_names)
            formulas.append(formula), times.append(elapsed_time), exp_accuracies.append(exp_accuracy)
            print(f"{class_to_explain} <-> {formula}")
            print("Elapsed time", elapsed_time)
            print("Explanation accuracy", exp_accuracy)

        methods.append("Relu")
        splits.append(seed)
        explanations.append(formulas[0])
        explanations_inv.append(formulas[1])
        model_accuracies.append(accuracy)
        explanation_accuracies.append(np.mean(exp_accuracies))
        explanation_accuracies_inv.append(np.mean(exp_accuracies))
        elapsed_times.append(np.mean(times))
        elapsed_times_inv.append(np.mean(times))

    results = pd.DataFrame({
        'method': methods,
        'split': splits,
        'explanation': explanations,
        'explanation_inv': explanations_inv,
        'model_accuracy': model_accuracies,
        'explanation_accuracy': explanation_accuracies,
        'explanation_accuracy_inv': explanation_accuracies_inv,
        'elapsed_time': elapsed_times,
        'elapsed_time_inv': elapsed_times_inv,
    })
    results_relu = results[results['method'] == "Relu"]
    results_relu.to_csv(os.path.join(results_dir, 'results_relu.csv'))
    print(results_relu)


    #%% md
    ## Summary
    #%%

    cols = ['model_accuracy', 'explanation_accuracy', 'explanation_accuracy_inv', 'elapsed_time', 'elapsed_time_inv']
    mean_cols = [f'{c}_mean' for c in cols]
    sem_cols = [f'{c}_sem' for c in cols]

    # general
    results_general = results[results['method'] == "General"]
    df_mean = results_general[cols].mean()
    df_sem = results_general[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_pruning = pd.concat([df_mean, df_sem])
    summary_pruning.name = 'general'

    # # lime
    # df_mean = results_lime[cols].mean()
    # df_sem = results_lime[cols].sem()
    # df_mean.columns = mean_cols
    # df_sem.columns = sem_cols
    # summary_lime = pd.concat([df_mean, df_sem])
    # summary_lime.name = 'lime'

    # relu
    results_relu = results[results['method'] == "Relu"]
    df_mean = results_relu[cols].mean()
    df_sem = results_relu[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_weights = pd.concat([df_mean, df_sem])
    summary_weights.name = 'relu'

    # psi
    results_psi = results[results['method'] == "Psi"]
    df_mean = results_psi[cols].mean()
    df_sem = results_psi[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_psi = pd.concat([df_mean, df_sem])
    summary_psi.name = 'psi'

    # tree
    results_tree = results[results['method'] == "Tree"]
    df_mean = results_tree[cols].mean()
    df_sem = results_tree[cols].sem()
    df_mean.columns = mean_cols
    df_sem.columns = sem_cols
    summary_tree = pd.concat([df_mean, df_sem])
    summary_tree.name = 'tree'

    summary = pd.concat([summary_pruning,
                         #                      summary_lime,
                         summary_weights,
                         summary_psi,
                         summary_tree], axis=1).T
    summary.columns = mean_cols + sem_cols
    print(summary)

    # %%

    summary.to_csv(os.path.join(results_dir, 'summary.csv'))

