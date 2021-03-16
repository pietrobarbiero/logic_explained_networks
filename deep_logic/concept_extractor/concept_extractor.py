import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from ..utils.metrics import Metric, TopkAccuracy
from ..models.base import BaseClassifier
from .cnn_models import RESNET10, get_model, CNN_MODELS, INCEPTION
from ..utils.base import NotAvailableError


class CNNConceptExtractor(BaseClassifier):
    """
    CNN classifier used for extracting concepts from images. It follows the strategy employed in Concept Bottleneck
    Models where a classifier (one of the interpretable in our case) is placed on top of a CNN working on images. The
    CNN provides for the low-level concepts, while the classifier provides for the final classification.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param loss: torch.nn.modules.loss
            type of loss to employ
        :param cnn_model: str
            one of the models implemented in the file cnn_models
        :param pretrained: bool
            whether to instantiate the model with the weights trained on ImageNet or randomly
    """

    def __init__(self, n_classes: int, cnn_model: str = RESNET10, loss=torch.nn.BCEWithLogitsLoss(),
                 transfer_learning: bool = False, pretrained: bool = False, name: str = "net",
                 device: torch.device = torch.device("cpu")):
        super().__init__(loss, name, device)

        assert cnn_model in CNN_MODELS, f"Required CNN model is not available, it needs to be among {CNN_MODELS.keys()}"
        assert not transfer_learning or pretrained, "Transfer learning can be applied only when pretrained is True"

        self.n_classes = n_classes
        self.cnn_model = cnn_model
        self.model = get_model(cnn_model, n_classes, pretrained=pretrained, transfer_learning=transfer_learning)
        self.pretrained = pretrained
        self.transfer_learning = transfer_learning
        self._output = None
        self._aux_output = None

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the InterpretableReLU is
        employed. An L1 weight regularization is also always applied

        :param outputs: output tensor from the forward function
        :param targets: label tensor
        :param kwargs: for compatibility
        :return: loss tensor value
        """
        loss = super().get_loss(outputs, targets)
        if self.cnn_model == INCEPTION and self._aux_output is not None:
            loss += 0.1 * super().get_loss(self._aux_output, targets)
        return loss

    def forward(self, x, logits=False) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :param logits: whether to return the logits or the probability value after the activation (default)
        :return: output classification
        """
        output = self.model(x)

        # Inception return 2 logits tensor
        if self.cnn_model == INCEPTION and self.training:
            self._aux_output = output[1]
            output = output[0]
        if logits:
            return output
        output = self.activation(output)
        return output

    def prune(self):
        NotAvailableError()

    def fit(self, train_set: Dataset, val_set: Dataset, batch_size: int = 32, epochs: int = 10, num_workers: int = 0,
            l_r: float = 0.01, lr_scheduler: bool = False, metric: Metric = TopkAccuracy(),
            device: torch.device = torch.device("cpu"), verbose: bool = True, **kwargs) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param train_set: training set on which to train
        :param val_set: validation set used for early stopping
        :param batch_size: number of training data for each step of the training
        :param epochs: number of epochs to train the model
        :param num_workers: number of process to employ when loading data
        :param l_r: learning rate parameter of the Adam optimizer
        :param lr_scheduler: whether to use learning rate scheduler (ReduceLROnPleteau)
        :param metric: metric to evaluate the predictions of the network
        :param device: device on which to perform the training
        :param verbose: whether to output or not epoch metrics
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Setting device
        self.to(device), self.train()

        # Setting loss function and optimizer
        if self.pretrained:
            fc_parameters = [param for param in self.model.fc.parameters()]
            parameters = [param for param in self.parameters() if param.requires_grad][:-2]
            if self.transfer_learning:
                params = [{'params': fc_parameters, 'lr': l_r}]
            else:
                params = [{'params': fc_parameters, 'lr': l_r},
                          {'params': parameters, 'lr': l_r * 1e-1}]
        else:
            params = self.parameters()
        optimizer = torch.optim.AdamW(params)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=verbose,
                                      factor=0.33, min_lr=1e-3*l_r) if lr_scheduler else None

        # Training epochs
        best_acc, best_epoch = 0.0, -1
        train_accs, val_accs, tot_losses = [], [], []
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=True,
                                                   num_workers=num_workers, prefetch_factor=4 if num_workers != 0 else 2)
        pbar = tqdm(range(epochs), ncols=100, position=0, leave=True) if verbose else None
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            tot_losses_i = []
            train_outputs, train_labels = [], []
            for i, data in enumerate(train_loader):
                # Load batch (dataset, labels) on the correct device
                batch_data, batch_labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # Network outputs on the current batch of dataset
                batch_outputs = self.forward(batch_data, logits=True)
                batch_activation = self.activation(batch_outputs)

                # Compute losses and update gradients
                tot_loss = self.get_loss(batch_outputs, batch_labels)
                tot_loss.backward()
                optimizer.step()
                # print(f"{i+1}/{len(train_loader)}, loss: {tot_loss:.4}")

                # Data moved to cpu again
                batch_outputs, batch_labels = batch_outputs.detach().cpu(), batch_labels.detach().cpu()
                tot_loss = tot_loss.detach().cpu()
                train_outputs.append(batch_activation), train_labels.append(batch_labels), tot_losses_i.append(tot_loss)

            train_outputs, train_labels = torch.cat(train_outputs), torch.cat(train_labels)
            tot_losses_i = torch.stack(tot_losses_i)

            # Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
            train_acc = self.evaluate(train_set, batch_size, metric, num_workers, device, train_outputs, train_labels)
            val_acc = self.evaluate(val_set, batch_size, metric, num_workers, device)

            # Step learning rate scheduler
            if lr_scheduler:
                scheduler.step(train_acc)

            # Save epoch results
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            tot_losses.append(tot_losses_i.mean().item())

            # Save best model
            if val_acc >= best_acc and epoch >= epochs / 2 or epochs == 1:
                best_acc = val_acc
                best_epoch = epoch + 1
                self.save()

            if verbose:
                pbar.set_postfix({"Tr_acc": f"{train_acc:.1f}", "Val_acc": f"{val_acc:.1f}",
                                  "Loss": f"{tot_losses[-1]:.3f}", "best_e": best_epoch})
                pbar.update()
                print(" ")

        if verbose:
            pbar.close()

        # Best model is loaded and saved again with buffer "trained" set to true
        self.load(device, set_trained=True)

        # Performance dictionary
        performance_dict = {
            "Tot losses": tot_losses,
            "Train accs": train_accs,
            "Val accs": val_accs,
            "Best epoch": best_epoch,
        }
        performance_df = pd.DataFrame(performance_dict)
        return performance_df
