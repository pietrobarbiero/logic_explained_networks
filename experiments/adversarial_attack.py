import os

import torch
from autoattack import AutoAttack
from torch.utils import data

from deep_logic.models.robust_cnn_classifier import RobustCNNClassifier
from deep_logic.utils.datasets import SingleLabelDataset


class AdversarialAttackError(BaseException):
    pass


class AdversarialAttackNotPerformedError(AdversarialAttackError):
    pass


class AdversarialAttackNotConsistentError(AdversarialAttackError):
    pass


def check_l2_adversarial_data_consistency(x_adv, x_test, epsilon):
    res = ((x_adv - x_test) ** 2).view(x_test.shape[0], -1).sum(-1).sqrt()
    if (res.max().item() - epsilon) / epsilon > 0.01:
        print(f"There was a problem in loading adv dataset, maximum perturbation {res.max().item()} exceeded "
              f"epsilon {epsilon}, by {(res.max().item() - epsilon) / epsilon}")
        raise AdversarialAttackNotConsistentError()
    else:
        print("Loaded adversarial data consistent")


def create_single_label_dataset(dataset: torch.utils.data.Dataset, main_classes: range, batch_size: int = 256):
    test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    x_test, y_test = [], []
    for (x, y) in test_loader:
        x_test.append(x), y_test.append(y)
    x_test, y_test = torch.cat(x_test, 0), torch.cat(y_test)
    y_test = y_test[:, main_classes].argmax(dim=1)
    single_label_dataset = SingleLabelDataset(x_test, y_test)
    return x_test, y_test, single_label_dataset


def single_label_evaluate(model: RobustCNNClassifier, dataset: SingleLabelDataset, main_classes: range,
                          reject: bool = False, adv: bool = False, batch_size: int = 128,
                          device: torch.device = torch.device("cpu")):
    model.eval()  # ALWAYS REMEMBER TO SET EVAL WHEN EVALUATING A RESNET
    model.to(device), model.model.to(device)
    outputs, labels, cons_losses, rejections = [], [], [], []
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        for i, loaded_data in enumerate(loader):
            batch_data, batch_labels = loaded_data[0].to(device), loaded_data[1].to(device)
            batch_output = model.forward(batch_data, logits=False)
            if reject:
                assert model.threshold is not None, "Threshold not calculated. self.calc_threshold need to be " \
                                                    "called before any forward operation when reject is set\n"
                cons_loss = model.constraint_loss(batch_output, sum_reduction=False)
                model.threshold = model.threshold.to(device=cons_loss.device)
                batch_rejections = cons_loss > model.threshold
                batch_output[batch_rejections == 1] = -1
            else:
                batch_rejections = torch.zeros(batch_output.shape[0])
            batch_output = batch_output[:, main_classes].argmax(dim=1)
            outputs.append(batch_output), labels.append(batch_labels), rejections.append(batch_rejections)

        outputs, labels = torch.cat(outputs, dim=0), torch.cat(labels, dim=0)
        rejections = torch.cat(rejections, dim=0)
        if adv:
            outputs[rejections == 1] = labels[rejections == 1]
        acc_single_label = labels.eq(outputs).sum().item() / outputs.shape[0] * 100
        rejection_rate = rejections.sum().item() / len(dataset) * 100
    model.set_eval_main_classes(False)
    model.train()
    return acc_single_label, rejection_rate


def load_adversarial_data(attack_path, load_sec_eval, attack, k_to_attack, seed, device, x_test, y_test, epsilon):
    if not (load_sec_eval and os.path.isfile(attack_path)):
        raise AdversarialAttackNotPerformedError()
    print(f"Attack {attack} against classifier constr {k_to_attack} seed {seed} already performed. "
          f"Loading saved data")
    adv_data = torch.load(attack_path, map_location=device)
    x_adv, y_adv = adv_data
    check_l2_adversarial_data_consistency(x_adv, x_test, epsilon)
    single_label_dataset_adv = SingleLabelDataset(x_adv, y_adv)
    return single_label_dataset_adv


def generate_adversarial_data(model: RobustCNNClassifier, dataset: SingleLabelDataset, dataset_name: str,
                              attack: str = "apgd-ce", epsilon: float = 0.5, batch_size: int = 128,
                              result_folder: str = ".", device: torch.device = torch.device("cpu")) \
        -> SingleLabelDataset:

    x_test, y_test = dataset.x, dataset.y
    attack_path = "attack_" + attack  + "_" + dataset_name + "_eps_" + str(epsilon)
    attack_path = os.path.join(result_folder, attack_path)
    print("Running attack " + attack + "...")

    if os.path.isfile(attack_path):
        x_adv, y_adv = torch.load(attack_path)
        print("Attack already performed")
    else:
        adversary = AutoAttack(model, norm='L2', eps=epsilon, device=device)
        model.eval()  # REMEMBER TO SET MODEL EVAL FOR RESNET BEFORE ATTACKING IT!
        model.set_eval_logits()
        model.set_eval_main_classes()
        adversary.attacks_to_run = [attack]
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
        model.set_eval_main_classes(False)
        model.set_eval_logits(False)
        print("Finished attack")
        y_adv = y_test
        torch.save((x_adv, y_adv), attack_path)

    single_label_dataset_adv = SingleLabelDataset(x_adv, y_adv)

    return single_label_dataset_adv

