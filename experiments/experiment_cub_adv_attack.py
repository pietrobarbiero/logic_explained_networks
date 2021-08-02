
if __name__ == "__main__":

    # %%

    import sys
    import time

    sys.path.append('..')
    import os
    import torch
    import pandas as pd
    from torch.nn import CrossEntropyLoss

    from deep_logic.models.relu_nn import XReluNN
    from deep_logic.models.psi_nn import PsiNetwork
    from deep_logic.utils.base import set_seed, ClassifierNotTrainedError, IncompatibleClassifierError
    from deep_logic.utils.metrics import Accuracy, F1Score
    from deep_logic.models.general_nn import XGeneralNN
    from deep_logic.utils.datasets import ConceptToTaskDataset, ImageToConceptAndTaskDataset
    from deep_logic.utils.data import get_splits_train_val_test, get_transform
    from deep_logic.concept_extractor.cnn_models import RESNET18
    from data import CUB200
    from data.download_cub import download_cub
    from experiments.CUB_200_2011.concept_extractor_cub import concept_extractor_cub
    from experiments.adversarial_attack import generate_adversarial_data, single_label_evaluate, \
        create_single_label_dataset

    results_dir = 'results/cub'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # %% md
    ## Loading CUB data
    # %%

    dataset_root = "../data/CUB_200_2011/"
    dataset_name = CUB200
    print(dataset_root)
    if not os.path.isdir(dataset_root):
        download_cub(dataset_root)
    else:
        print("Dataset already downloaded")

    #%% md
    ## Defining dataset
    #%%

    dataset = ConceptToTaskDataset(dataset_root)
    train_data, val_data, test_data = get_splits_train_val_test(dataset, load=False)
    class_names = dataset.classes
    concept_names = dataset.attribute_names
    print("Concept names", concept_names)
    n_features = dataset.n_attributes
    print("Number of attributes", n_features)
    n_classes = dataset.n_classes
    print("Number of classes", n_classes)
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    #%% md
    ## Define loss, metrics and methods
    #%%

    loss = CrossEntropyLoss()
    metric = Accuracy()
    expl_metric = F1Score()
    method_list = ['General', 'Psi', 'Relu']
    print("Methods", method_list)
    batch_size = 128
    attack = "apgd-t"

    # %% md
    ## Training black box on images (to predict labels and attributes)
    # %%

    bb_model = concept_extractor_cub(dataset_root, result_folder="CUB_200_2011", robust=True,
                                     multi_label=True, device=device, cnn_model=RESNET18,
                                     seeds=[0], save_predictions=False, show_image=False)[0]

    #%% md
    ## Using outputs of black box model as input and targets for other surrogate models
    #%%

    train_transform = get_transform(dataset=dataset_name, data_augmentation=True)
    bb_dataset = ImageToConceptAndTaskDataset(dataset_root, train_transform, dataset_name=dataset_name)
    train_data_bb, val_data_bb, test_data_bb = get_splits_train_val_test(bb_dataset, load=False)
    prediction_name = os.path.join(results_dir, "black_box_predictions.pth")
    if os.path.isfile(prediction_name):
        outputs_bb, labels_bb = torch.load(prediction_name)
    else:
        with torch.no_grad():
            outputs_bb, labels_bb = bb_model.predict(bb_dataset, device=device, batch_size=batch_size)
        torch.save((outputs_bb, labels_bb), prediction_name)
    dataset.targets = outputs_bb[:, :n_classes].detach().cpu()
    dataset.attributes = outputs_bb[:, n_classes:].detach().cpu()
    print("Black Box predictions saved")

    #%% md
    ## Attacking model
    #%%
    _, _, test_data_bb_sl = create_single_label_dataset(test_data_bb, range(n_classes))
    bb_accuracy_clean, _ = single_label_evaluate(bb_model, test_data_bb_sl, range(n_classes), device=device)
    print("Main classes accuracy on clean test data", bb_accuracy_clean)
    adv_dataset = generate_adversarial_data(bb_model, test_data_bb_sl, dataset_name, attack,
                                            result_folder=results_dir, device=device)
    bb_accuracy_adv, bb_rejection_adv = single_label_evaluate(bb_model, adv_dataset, range(n_classes), device=device)
    print("Main classes accuracy on adv test data", bb_accuracy_adv)
    multi_label_test_labels = labels_bb[test_data_bb.indices, :]
    with torch.no_grad():
        adv_multilabel_prediction, _ = bb_model.predict(adv_dataset, batch_size, device=device)
    accuracy_adv_data_multilabel = bb_model.evaluate(adv_dataset, outputs=adv_multilabel_prediction,
                                                     labels=multi_label_test_labels, metric=F1Score())
    print("Multilabel accuracy on adv test data", accuracy_adv_data_multilabel)

    #%% md
    ## Setting training hyperparameters
    #%%

    epochs = 1000
    n_processes = 1
    timeout = 6 * 60 * 60  # 6 h timeout
    l_r = 1e-3
    lr_scheduler = False
    top_k_explanations = None
    simplify = True
    seeds = [*range(0, 5)]
    print("Seeds", seeds)

    #%% md
    ## Training explanation methods and testing explanations on the adversarial data
    #%%

    for method in method_list:

        methods = []
        splits = []
        model_accuracies = []
        model_explanations = []
        rejection_rates = []
        bb_accuracy_with_rej_clean = []
        bb_accuracy_with_rej_adv = []
        bb_rejection_rate_clean = []
        bb_rejection_rate_adv = []

        for seed in seeds:
            set_seed(seed)
            name = os.path.join(results_dir, f"{method}_{seed}")

            train_data, val_data, test_data = get_splits_train_val_test(dataset, load=False)
            x_train = torch.as_tensor(dataset.attributes[train_data.indices])
            y_train = torch.as_tensor(dataset.targets[train_data.indices])
            x_val = torch.as_tensor(dataset.attributes[val_data.indices])
            y_val = torch.as_tensor(dataset.targets[val_data.indices])
            x_test = torch.as_tensor(dataset.attributes[test_data.indices])
            y_test = torch.as_tensor(dataset.targets[test_data.indices])
            print(train_data.indices)

            # Setting device
            print(f"Training {name} classifier...")
            start_time = time.time()

            if method == 'Psi':
                # Network structures
                l1_weight = 1e-4
                hidden_neurons = [10]
                fan_in = 4
                model = PsiNetwork(n_classes, n_features, hidden_neurons, loss,
                                   l1_weight, name=name, fan_in=fan_in)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas = []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(i, concept_names, simplify=simplify)
                    formulas.append(formula)

            elif method == 'General':
                # Network structures
                lr_general = l_r
                l1_weight = 1e-4
                hidden_neurons = [20]
                set_seed(seed)
                model = XGeneralNN(n_classes=dataset.n_classes, n_features=n_features, hidden_neurons=hidden_neurons,
                                   loss=loss, l1_weight=l1_weight, fan_in=10, name=name)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=lr_general, metric=metric,
                                        lr_scheduler=lr_scheduler, device=device, save=True, verbose=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=F1Score(), outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas = []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_train, y_train, i, top_k_explanations, concept_names,
                                                           simplify=simplify, metric=expl_metric,
                                                           x_val=x_val, y_val=y_val )
                    formulas.append(formula)

            elif method == 'Relu':
                # Network structures
                l1_weight = 1e-7
                hidden_neurons = [300, 200]
                print("L1 weight", l1_weight)
                print("Hidden neurons", hidden_neurons)
                model = XReluNN(n_classes, n_features, hidden_neurons, loss, name=name,
                                l1_weight=l1_weight)
                try:
                    model.load(device)
                    print(f"Model {name} already trained")
                except (ClassifierNotTrainedError, IncompatibleClassifierError):
                    results = model.fit(train_data, val_data, epochs=epochs, l_r=l_r, verbose=True,
                                        metric=metric, lr_scheduler=lr_scheduler, device=device, save=True)
                outputs, labels = model.predict(test_data, device=device)
                accuracy = model.evaluate(test_data, metric=metric, outputs=outputs, labels=labels)
                print("Test model accuracy", accuracy)
                formulas = []
                for i, class_to_explain in enumerate(dataset.classes):
                    formula = model.get_global_explanation(x_train, y_train, i, top_k_explanations, concept_names,
                                                           simplify=simplify, metric=expl_metric,
                                                           x_val=x_val, y_val=y_val)
                    formulas.append(formula)

            else:
                raise NotImplementedError(f"{method} not implemented")

            bb_model.set_explanations(formulas)
            bb_model.calc_threshold(val_data_bb, batch_size=batch_size)
            bb_accuracy, bb_rejection_rate = single_label_evaluate(bb_model, test_data_bb_sl, range(n_classes),
                                                                   reject=True, adv=False, device=device)
            print("Accuracy and rejection on clean data", bb_accuracy, bb_rejection_rate)
            adv_bb_accuracy, bb_adv_rejection_rate = single_label_evaluate(bb_model, adv_dataset, range(n_classes),
                                                                           reject=True, adv=True, device=device)
            print("Accuracy and rejection on adv data", adv_bb_accuracy, bb_adv_rejection_rate)

            methods.append(method)
            splits.append(seed)
            model_explanations.append(formulas[0])
            model_accuracies.append(accuracy)
            bb_accuracy_with_rej_clean.append(bb_accuracy)
            bb_accuracy_with_rej_adv.append(adv_bb_accuracy)
            bb_rejection_rate_clean.append(bb_rejection_rate)
            bb_rejection_rate_adv.append(bb_adv_rejection_rate)

        results = pd.DataFrame({
            'method': methods,
            'split': splits,
            'explanation': model_explanations,
            'model_accuracy': model_accuracies,
            'bb_accuracy_clean': [bb_accuracy_clean] * len(seeds),
            'bb_accuracy_adv': [bb_accuracy_adv] * len(seeds),
            'bb_accuracy_clean_rej': bb_accuracy_with_rej_clean,
            'bb_accuracy_adv_rej': bb_accuracy_with_rej_adv,
            'bb_rejection_rate_clean': bb_rejection_rate_clean,
            'bb_rejection_rate_adv': bb_rejection_rate_adv,
        })
        results.to_csv(os.path.join(results_dir, f'adv_results_{method}.csv'))
        print(results)

    results_df = {}
    summaries = {}
    for m in method_list:
        results_df[m] = pd.read_csv(os.path.join(results_dir, f"adv_results_{m}.csv"))

    results_df = pd.concat([results_df[method] for method in method_list])
    results_df.to_csv(os.path.join(results_dir, f'adv_results.csv'))
