from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class Classifier(torch.nn.Module, ABC):
	"""
		Classifier is an abstract class representing a generic classifier
		init() and forward() methods are required to be implemented by extending classes

		Parameters
		----------
		name: str
			name of the network: used for saving and loading purposes
	 """

	@abstractmethod
	def __init__(self, name: str = "net",):

		super(Classifier, self).__init__()
		self.name = name
		self.register_buffer("trained", torch.tensor(False))

	@abstractmethod
	def forward(self, x):
		assert not torch.isnan(x).any(), "Input data contain nan values"
		assert not torch.isinf(x).any(), "Input data contain inf values"

	@abstractmethod
	def get_loss(self):
		pass

	def to(self, *args, **kwargs):
		self.model.to(*args, **kwargs)
		super().to(*args, **kwargs)

	@staticmethod
	def set_seed(seed):
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	def get_device(self):
		return [*self.model.parameters()][0].device

	def fit(self, train_set: Dataset, val_set: Dataset, batch_size: int = 16, epochs: int = 170,
	        l_r: float = 0.1, device: str = "cpu", verbose: bool = True):

		# Check if model is already trained
		try:
			self.load(device=device)
			print("Model already trained. Skipping training...")
			return None
		except ClassifierNotTrainedError:
			print("Training model...")

		# Setting device
		device = torch.device(device)
		self.to(device), self.train()

		# Setting loss function and optimizer
		optimizer = torch.optim.Adam(self.parameters(), lr=l_r)
		loss = self.get_loss()

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
				tot_loss = loss(self, batch_outputs, batch_labels, train_set, device)
				tot_loss.backward()
				optimizer.step()

				# Data moved to cpu again
				batch_outputs, batch_labels = batch_outputs.detach().cpu(), batch_labels.detach().cpu()
				tot_loss = tot_loss.detach().cpu()
				train_outputs.append(batch_outputs), train_labels.append(batch_labels), tot_losses.append(tot_loss)

			train_outputs, train_labels = torch.cat(train_outputs), torch.cat(train_labels)
			tot_losses = torch.cat(tot_losses)

			# Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
			train_acc = self.evaluate(train_set, batch_size, device, train_outputs, train_labels)
			val_acc = self.evaluate(val_set, batch_size, device)

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

	def evaluate(self, dataset, batch_size, device, outputs=None, labels=None, attrs=None, attrs_labels=None):
		self.eval(), self.to(device)
		with torch.no_grad():
			if outputs is None or labels is None or attrs is None or attrs_labels is None:
				outputs, labels = self.predict(dataset, batch_size, device)
			acc5 = self.top_k_accuracy(outputs, labels)
		self.train()
		return acc5

	def predict(self, dataset, batch_size, device):
		outputs, labels, attrs, attrs_labels = [], [], [], []
		loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)
		for i, data in enumerate(loader):
			batch_data, batch_labels, = data[0].to(device), data[1].to(device)
			batch_outputs = self.forward(batch_data)
			outputs.append(batch_outputs)
			labels.append(batch_labels)
		return torch.cat(outputs), torch.cat(labels)

	@staticmethod
	def top_k_accuracy(outputs, labels, k=5):
		labels = labels.argmax(dim=1)
		n_samples = labels.shape[0]
		_, topk_outputs = outputs.topk(k, 1)
		topk_acc = topk_outputs.eq(labels.reshape(-1, 1)).sum() / n_samples * 100
		return topk_acc.cpu().item()

	@staticmethod
	def f1(outputs: torch.Tensor, labels: torch.Tensor):
		sigmoid_output = torch.sigmoid(outputs).cpu().numpy()
		discrete_output = sigmoid_output > 0.5
		f1_val = f1_score(discrete_output, labels.cpu().numpy(), average='macro', zero_division=0) * 100
		return f1_val

	def save(self, set_trained=False, name=None):
		if name is None:
			name = self.name
		if set_trained:
			self._set_trained()
		torch.save(self.state_dict(), name)

	def load(self, device, set_trained=False, name=None):
		if name is None:
			name = self.name
		try:
			incompatible_keys = self.load_state_dict(torch.load(name,
			                                                    map_location=torch.device(device)), strict=False)
		except FileNotFoundError:
			raise ClassifierNotTrainedError() from None
		else:
			if len(incompatible_keys.missing_keys) > 1 or len(incompatible_keys.unexpected_keys) > 0:
				raise IncompatibleClassifierError(incompatible_keys.missing_keys, incompatible_keys.unexpected_keys)
		if set_trained or (len(incompatible_keys.missing_keys) > 0 and incompatible_keys.missing_keys[0] == 'trained'):
			self.save(set_trained=True)
		if not self.trained:
			raise ClassifierNotTrainedError()

	def _set_trained(self):
		self.trained = torch.tensor(True)


class ClassifierNotTrainedError(Exception):
	def __init__(self):
		self.message = "Classifier not trained"

	def __str__(self):
		return self.message


class IncompatibleClassifierError(Exception):
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
