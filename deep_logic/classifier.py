from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from metrics import Metric, TopkAccuracy


class Classifier(torch.nn.Module, ABC):
	"""
		Classifier is an abstract class representing a generic classifier.
		init(), forward() and get_loss() methods are required to be implemented by extending classes

		:param name: str
			name of the network: used for saving and loading purposes
		:var trained: bool
			flag set at the end of the training and saved with the model. Only trained model can be loaded from memory
	 """

	@abstractmethod
	def __init__(self, name: str = "net",):

		super(Classifier, self).__init__()
		self.name = name
		self.register_buffer("trained", torch.tensor(False))

	@abstractmethod
	def forward(self, x: torch.Tensor):
		"""
		forward method that needs to be implemented by each extending class. In this class only some checks on the input
		are performed.
		:param x: input tensor
		"""
		assert not torch.isnan(x).any(), "Input data contain nan values"
		assert not torch.isinf(x).any(), "Input data contain inf values"

	@abstractmethod
	def get_loss(self, output: torch.Tensor, target: torch.Tensor):
		"""
		get_loss method is used by each class to calculate its own loss according to the different training strategy
		"""
		pass

	@staticmethod
	def set_seed(seed):
		"""
		Static method used to set the seed for an experiment. Needs to be called before doing anything else.
		:param seed:
		"""
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def get_device(self) -> torch.device:
		"""
		Return the device on which the classifier is actually loaded
		:return: device in use
		"""
		return [*self.model.parameters()][0].device

	def fit(self, train_set: Dataset, val_set: Dataset, batch_size: int = 16, epochs: int = 170,
	        l_r: float = 0.1, metric: Metric = TopkAccuracy(), device: torch.device = torch.device("cpu"),
	        verbose: bool = True) -> pd.DataFrame:
		"""
		fit function that execute many of the common operation generally performed by many method during training.
		Adam optimizer is always employed

		:param train_set: training set on which to train
		:param val_set: validation set used for early stopping
		:param batch_size: number of training data for each step of the training
		:param epochs: number of epochs to train the model
		:param l_r: learning rate parameter of the Adam optimizer
		:param metric: metric to evaluate the predictions of the network
		:param device: device on which to perform the training
		:param verbose: whether to output or not epoch metrics
		:return: pandas dataframe collecting the metrics from each epoch
		"""

		# Setting device
		device = torch.device(device)
		self.to(device), self.train()

		# Setting loss function and optimizer
		optimizer = torch.optim.Adam(self.parameters(), lr=l_r)

		# Training epochs
		best_acc, best_epoch = 0.0, 0
		train_accs, val_accs, tot_losses = [], [], []
		train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, num_workers=8)
		pbar = tqdm(range(epochs), ncols=100, position=0, leave=True) if verbose else None
		torch.autograd.set_detect_anomaly(True)
		for epoch in range(epochs):
			if verbose:
				pbar.set_postfix({"Epoch": f"{epoch + 1}/{epochs}"})

			train_outputs, train_labels = [], []
			for data in train_loader:
				# Load batch (dataset, labels) on the correct device
				batch_data, batch_labels = data[0].to(device), data[1].to(device)
				optimizer.zero_grad()

				# Network outputs on the current batch of dataset
				batch_outputs = self.forward(batch_data)

				# Compute losses and update gradients
				tot_loss = self.get_loss(batch_outputs, batch_labels)
				tot_loss.backward()
				optimizer.step()

				# Data moved to cpu again
				batch_outputs, batch_labels = batch_outputs.detach().cpu(), batch_labels.detach().cpu()
				tot_loss = tot_loss.detach().cpu()
				train_outputs.append(batch_outputs), train_labels.append(batch_labels), tot_losses.append(tot_loss)

			train_outputs, train_labels = torch.cat(train_outputs), torch.cat(train_labels)
			tot_losses = torch.cat(tot_losses)

			# Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
			train_acc = self.evaluate(train_set, batch_size, metric, device, train_outputs, train_labels)
			val_acc = self.evaluate(val_set, batch_size, metric, device)

			# Save epoch results
			train_accs.append(train_acc)
			val_accs.append(val_acc)
			tot_losses.append(tot_losses.mean().item())

			if verbose:
				pbar.set_postfix({"Train_acc": f"{train_acc:.1f}", "Val_acc": f"{val_acc:.1f}",
				                  "best_epoch": best_epoch})
				pbar.update()

			# Save best model
			if val_acc >= best_acc and epoch >= epochs / 2:
				best_acc = val_acc
				best_epoch = epoch + 1
				self.save(set_trained=True)

		# Best model is loaded and saved again with buffer "trained" set to true
		self.load(device, set_trained=True)

		# Performance dictionary
		performance_dict = {
			"tot_loss": tot_losses,
			"train_accs": train_accs,
			"val_accs": val_accs,
			"best_epoch": best_epoch,
		}
		performance_df = pd.DataFrame(performance_dict)
		return performance_df

	def evaluate(self, dataset: Dataset, batch_size: int, metric: Metric = TopkAccuracy(),
	             device: torch.device = torch.device("cpu"), outputs=None, labels=None) -> float:
		"""
		Evaluate function to test without training the performance of the model on a certain dataset
		:param dataset: dataset on which to test
		:param batch_size: number of training data for each step of the training
		:param metric: metric to evaluate the predictions of the network
		:param device: device on which to perform the training
		:param outputs: if the output is passed is not calculated again
		:param labels: to be passed together with the output
		:return: metric evaluated on the dataset
		"""
		self.eval(), self.to(device)
		with torch.no_grad():
			if outputs is None or labels is None:
				outputs, labels = self.predict(dataset, batch_size, device)
			metric_val = metric(outputs, labels)
		self.train()
		return metric_val

	def predict(self, dataset, batch_size, device) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Predict function to compute the prediction of the model on a certain dataset
		:param dataset: dataset on which to test
		:param batch_size: number of training data for each step of the training
		:param device: device on which to perform the training
		:return: a tuple containing the outputs computed on the dataset and the labels
		"""
		outputs, labels, attrs, attrs_labels = [], [], [], []
		loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)
		for i, data in enumerate(loader):
			batch_data, batch_labels, = data[0].to(device), data[1].to(device)
			batch_outputs = self.forward(batch_data)
			outputs.append(batch_outputs)
			labels.append(batch_labels)
		return torch.cat(outputs), torch.cat(labels)

	def save(self, set_trained=False, name=None) -> None:
		"""
		Save model on a file named with the name of the model if parameter name is not set.
		:param set_trained: Used to set the buffer flag "trained" at the end of training.
		:param name: Save the model with a name different from the one assigned in the __init__
		"""
		if name is None:
			name = self.name
		if set_trained:
			self._set_trained()
		torch.save(self.state_dict(), name)

	def load(self, device, set_trained=False, name=None) -> None:
		"""
		Load model on a specific device (can be different from the one used during training). If set_trained is true than
		the model flag "trained" is set to true first and the model is saved again. If set_trained is not set and
		the model flag "trained" is not true a ClassifierNotTrainedError is raised
		:param device: device on which to load the model
		:param set_trained: whether to set the buffer flag "trained" before loading or not.
		:param name: Load a model with a name different from the one assigned in the __init__
		"""
		if name is None:
			name = self.name
		try:
			incompatible_keys = self.load_state_dict(torch.load(name,
			                                                    map_location=torch.device(device)), strict=False)
		except FileNotFoundError:
			raise ClassifierNotTrainedError() from None
		else:
			if len(incompatible_keys.missing_keys) > 0 or len(incompatible_keys.unexpected_keys) > 0:
				raise IncompatibleClassifierError(incompatible_keys.missing_keys, incompatible_keys.unexpected_keys)
		if set_trained:
			self.save(set_trained=True)
		if not self.trained:
			raise ClassifierNotTrainedError()

	def _set_trained(self) -> None:
		"""
		Internal function used to set the buffer "trained" to true
		"""
		self.trained = torch.tensor(True)


class ClassifierNotTrainedError(Exception):
	"""
	Error raised when we try to load a classifier that it does not exists or when the classifier exists but
	its training has not finished.
	"""
	def __init__(self):
		self.message = "Classifier not trained"

	def __str__(self):
		return self.message


class IncompatibleClassifierError(Exception):
	"""
	Error raised when we try to load a classifier with a different structure with respect to the current model.
	"""
	def __init__(self, missing_keys, unexpected_keys):
		self.message = "Unable to load the selected classifier.\n"
		for key in missing_keys:
			self.message += "Missing key: " + str(key) + ".\n"
		for key in unexpected_keys:
			self.message += "Unexpected key: " + str(key) + ".\n"

	def __str__(self):
		return self.message


if __name__ == "__main__":
	pass
	# Classifier.set_seed(0)
	# d_path = "..//data//CUB_200_2011"
	# d = dl.CUB200
	# cosine = False
	# os.environ['CUDA_VISIBLE_DEVICES'] = '0,'  # '1'
	# dev = "cpu" if not torch.cuda.is_available() else "cuda"
	# n = "cosine.pth" if cosine else "linear.pth"
	# attr_w = 25
	# orth_w = 0.0025
	# wd = 0.0001
	# lr = 0.1
	# c_w = 0.01  # 0.001
	#
	# fsc_dataset = dl.FSCDataset(d_path, d)
	# train_d, fine_t_d = fsc_dataset.get_splits_for_fsc()
	# train_d, val_d = train_d.get_splits_train_val()
	#
	# clf = Classifier(d, len(train_d.classes), name=n, use_attributes=False, cosine_classifier=cosine)
	#
	# df: pd.DataFrame = clf.fit(train_d, val_d, device=dev, verbose=True, l_r=lr, weight_decay=wd)
	# # df : pd.DataFrame = pd.read_csv(f"{n}.csv")
	#
	# tm = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
	# df.to_csv(f"{n}_{tm}.csv")
	# print("Dataframe_saved")
